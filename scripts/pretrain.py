#!/usr/bin/env python3
"""
Stage 0: pretrain the DiT backbone on all users mixed (no per-user bias).

Trains all DiT parameters with flow-matching MSE loss and user_id=None.
Saves backbone.pt at each checkpoint — loaded by scripts/train.py for Stage 1.

Usage:
    PYTHONPATH="$(pwd)" uv run accelerate launch --num_processes=1 \\
        scripts/pretrain.py \\
        --config configs/dp_pretrain.yaml \\
        --data_dir data/ \\
        --output_dir outputs/dp_ucvla/pretrain
"""

import argparse
import logging
import os
from datetime import datetime

import torch
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm
import wandb

from datasets import get_train_dataset, get_val_dataset, collate_fn
from models.clip_encoder import CLIPEncoder
from models.dp.model import UCVLADiT
from models.dp.runner import UCVLADPRunner

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/dp_pretrain.yaml")
    p.add_argument("--data_dir", type=str, default="data/")
    p.add_argument("--output_dir", type=str, default="outputs/dp_ucvla/pretrain")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--mixed_precision", type=str, default="bf16",
                   choices=["no", "fp16", "bf16"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_config=ProjectConfiguration(project_dir=args.output_dir),
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    logging.getLogger().setLevel(logging.WARNING)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        log_dir = os.path.join("logs", "dp_pretrain")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{timestamp}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        logging.getLogger().addHandler(fh)
        logging.getLogger().setLevel(logging.INFO)
        print(f"Logging to {log_file}")

    # ------------------------------------------------------------------ #
    # Build model                                                          #
    # ------------------------------------------------------------------ #
    clip = CLIPEncoder()
    clip_transform = clip.get_transform()
    clip.to(accelerator.device)

    model = UCVLADiT(
        n_users=cfg["n_users"],
        bias_dim=cfg["bias_dim"],
        hidden_size=cfg["hidden_size"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        pred_horizon=cfg["pred_horizon"],
        action_dim=cfg["action_dim"],
        state_dim=cfg["state_dim"],
        clip_dim=cfg.get("clip_dim", 768),
    )
    runner = UCVLADPRunner(
        model=model,
        lambda_triplet=0.0,
        lambda_ortho=0.0,
    )
    # Do NOT freeze — train all DiT params

    n_trainable = sum(p.numel() for p in runner.model.parameters())
    logger.info(f"Trainable params: {n_trainable:,}  (full DiT backbone)")

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    shards_dir = os.path.join(args.data_dir, "mug_handover_webdataset")
    train_ds = get_train_dataset(shards_dir, clip_transform, cfg["state_dim"])
    val_ds = get_val_dataset(shards_dir, clip_transform, cfg["state_dim"])

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ------------------------------------------------------------------ #
    # Optimiser + scheduler                                                #
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.AdamW(
        runner.model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    warmup = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=cfg["warmup_steps"])
    cosine = CosineAnnealingLR(optimizer, T_max=cfg["max_steps"] - cfg["warmup_steps"])
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg["warmup_steps"]])

    runner, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        runner, optimizer, train_loader, val_loader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(cfg.get("wandb_project", "ucvla-proxy-pretrain"), config=cfg)

    # ------------------------------------------------------------------ #
    # Resume                                                               #
    # ------------------------------------------------------------------ #
    global_step = 0
    if args.resume_from_checkpoint:
        ckpt_path = args.resume_from_checkpoint
        if ckpt_path == "latest":
            dirs = sorted(
                [d for d in os.listdir(args.output_dir) if d.startswith("step-")],
                key=lambda x: int(x.split("-")[1]),
            )
            ckpt_path = os.path.join(args.output_dir, dirs[-1], "backbone.pt") if dirs else None

        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            accelerator.unwrap_model(runner).model.load_state_dict(ckpt["model"])
            global_step = ckpt.get("global_step", 0)
            logger.info(f"Resumed from {ckpt_path} at step {global_step}")

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    def save_checkpoint(step: int) -> None:
        if not accelerator.is_main_process:
            return
        save_dir = os.path.join(args.output_dir, f"step-{step}")
        os.makedirs(save_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(runner)
        torch.save(
            {
                "model": unwrapped.model.state_dict(),
                "global_step": step,
            },
            os.path.join(save_dir, "backbone.pt"),
        )
        logger.info(f"Saved backbone checkpoint at step {step}")

    def run_val(step: int) -> None:
        runner.eval()
        val_losses: list[float] = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(accelerator.device)
                actions = batch["actions"].to(accelerator.device, dtype=torch.float32)
                states = batch["states"].to(accelerator.device, dtype=torch.float32)

                clip_tokens = clip(images)
                loss, _ = accelerator.unwrap_model(runner).compute_loss(
                    action_gt=actions,
                    clip_tokens=clip_tokens,
                    state=states,
                    user_id=None,
                )
                val_losses.append(loss.item())

        if accelerator.is_main_process and val_losses:
            mean_val = sum(val_losses) / len(val_losses)
            logger.info(f"step {step:>6}  val_loss={mean_val:.4f}")
            accelerator.log({"val/loss": mean_val}, step=step)

        runner.train()

    progress_bar = tqdm(
        range(global_step, cfg["max_steps"]),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Pretrain")

    runner.train()
    for batch in train_loader:
        if global_step >= cfg["max_steps"]:
            break

        images = batch["images"].to(accelerator.device)
        actions = batch["actions"].to(accelerator.device, dtype=torch.float32)
        states = batch["states"].to(accelerator.device, dtype=torch.float32)

        with torch.no_grad():
            clip_tokens = clip(images)

        with accelerator.accumulate(runner):
            loss, log_dict = accelerator.unwrap_model(runner).compute_loss(
                action_gt=actions,
                clip_tokens=clip_tokens,
                state=states,
                user_id=None,
            )
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        global_step += 1
        progress_bar.update(1)

        log_dict["lr"] = lr_scheduler.get_last_lr()[0]

        if accelerator.is_main_process and global_step % cfg["log_every"] == 0:
            accelerator.log(log_dict, step=global_step)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            lr = log_dict["lr"]
            mse = log_dict.get("loss/mse", loss.item())
            logger.info(f"step {global_step:>6}  loss={loss.item():.4f}  mse={mse:.4f}  lr={lr:.2e}")

        if global_step % cfg["val_every"] == 0:
            run_val(global_step)

        if global_step % cfg["save_every"] == 0:
            save_checkpoint(global_step)

    save_checkpoint(global_step)
    accelerator.end_training()
    logger.info("Pretraining complete.")


if __name__ == "__main__":
    main()
