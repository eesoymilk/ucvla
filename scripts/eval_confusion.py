#!/usr/bin/env python3
"""
Cross-user confusion matrix for UCVLA proxy (DiT diffusion policy).

For each validation sample from user u, predict with all user biases.
The correct bias should produce lower action reconstruction error.

Adapted from RDT2's scripts/eval_ucvla.py — VLM removed, CLIP encoder used.

Usage:
    PYTHONPATH="$(pwd)" uv run python scripts/eval_confusion.py \\
        --weights outputs/dp_ucvla/stage1/step-10000/ucvla_weights.pt \\
        --config  configs/dp_ucvla.yaml \\
        --data_dir data/
"""

import argparse
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
import yaml

from datasets import get_val_dataset, collate_fn
from models.clip_encoder import CLIPEncoder
from models.dp.model import UCVLADiT
from models.dp.runner import UCVLADPRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True,
                   help="Path to ucvla_weights.pt checkpoint")
    p.add_argument("--config", type=str, default="configs/dp_ucvla.yaml")
    p.add_argument("--data_dir", type=str, default="data/")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------ #
    # Load model                                                           #
    # ------------------------------------------------------------------ #
    print("Loading CLIP encoder...")
    clip = CLIPEncoder().to(device)
    clip_transform = clip.get_transform()

    ckpt = torch.load(args.weights, map_location="cpu")
    n_users = ckpt.get("n_users", cfg["n_users"])
    bias_dim = ckpt.get("bias_dim", cfg["bias_dim"])

    print(f"Loading UCVLADiT  (n_users={n_users}, bias_dim={bias_dim})...")
    model = UCVLADiT(
        n_users=n_users,
        bias_dim=bias_dim,
        hidden_size=cfg["hidden_size"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        pred_horizon=cfg["pred_horizon"],
        action_dim=cfg["action_dim"],
        state_dim=cfg["state_dim"],
    )
    runner = UCVLADPRunner(model=model)
    model.user_bias.load_state_dict(ckpt["user_bias"])
    model.bias_proj.load_state_dict(ckpt["bias_proj"])
    runner.eval().to(device)

    # ------------------------------------------------------------------ #
    # Validation data                                                      #
    # ------------------------------------------------------------------ #
    shards_dir = os.path.join(args.data_dir, "mug_handover_webdataset")
    val_ds = get_val_dataset(shards_dir, clip_transform, cfg["state_dim"])
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # ------------------------------------------------------------------ #
    # Confusion matrix                                                     #
    # ------------------------------------------------------------------ #
    # errors[(uid_true, uid_pred)] = [mse_per_sample, ...]
    errors: dict[tuple[int, int], list[float]] = defaultdict(list)

    print(f"\nRunning cross-user eval (n_users={n_users})...")
    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            actions = batch["actions"].to(device, dtype=dtype)
            states = batch["states"].to(device, dtype=dtype)
            uid_true = batch["user_ids"].to(device)

            clip_tokens = clip(images)  # (B, 196, 512)

            for uid_pred in range(n_users):
                uid_tensor = torch.full_like(uid_true, uid_pred)
                pred = runner.predict_action(
                    clip_tokens=clip_tokens,
                    state=states,
                    user_id=uid_tensor,
                )
                # Per-sample MSE
                err = F.mse_loss(pred, actions, reduction="none").mean(dim=[1, 2])
                for i, ut in enumerate(uid_true.tolist()):
                    errors[(ut, uid_pred)].append(err[i].item())

    # ------------------------------------------------------------------ #
    # Print results                                                        #
    # ------------------------------------------------------------------ #
    print("\n── Cross-user confusion matrix (MSE, lower=better) ──────────────")
    print("   rows=true user, cols=predicted bias\n")
    header = "        " + "  ".join(f"bias_{j}" for j in range(n_users))
    print(header)
    for ut in range(n_users):
        row = []
        for up in range(n_users):
            vals = errors[(ut, up)]
            mean = sum(vals) / len(vals) if vals else float("nan")
            marker = " ←" if up == ut else "   "
            row.append(f"{mean:.5f}{marker}")
        print(f"  user_{ut}: " + "  ".join(row))

    print("\n── Diagonal dominance check ──────────────────────────────────────")
    wins = 0
    for ut in range(n_users):
        diag_vals = errors[(ut, ut)]
        if not diag_vals:
            print(f"  user_{ut}: no samples")
            continue
        correct = sum(diag_vals) / len(diag_vals)
        others = [
            sum(errors[(ut, up)]) / len(errors[(ut, up)])
            for up in range(n_users)
            if up != ut and errors[(ut, up)]
        ]
        if others and correct < min(others):
            wins += 1
            print(f"  user_{ut}: correct bias WINS ✓  ({correct:.5f} < {min(others):.5f})")
        else:
            best_other = min(others) if others else float("nan")
            print(f"  user_{ut}: correct bias LOSES ✗  ({correct:.5f} vs min_other={best_other:.5f})")

    print(f"\n  {wins}/{n_users} users have correct bias winning")
    print("\n── Success criterion ─────────────────────────────────────────────")
    if wins == n_users:
        print("  PASS — all users win on correct bias → port to RDT2")
    else:
        print("  FAIL — not all users win; try different loss ablation")


if __name__ == "__main__":
    main()
