#!/usr/bin/env python3
"""
Behavioral divergence eval for UCVLA proxy.

For the same input image, generate actions under each user bias and measure
how much predictions diverge across users. This is a necessary condition for
the embeddings to encode preferences: if pred_i ≈ pred_j everywhere, no MSE
eval can show diagonal dominance regardless of cosim.

Reports:
  1. Mean pairwise L2 divergence per (i, j) — global magnitude of effect
  2. Divergence bucketed by chunk_weight (low / mid / high) — does the effect
     concentrate in discriminative frames?

Usage:
    PYTHONPATH="$(pwd)" uv run python scripts/eval_divergence.py \\
        --weights      outputs/dp_ucvla/stage1_10d/step-50000/ucvla_weights.pt \\
        --chunk_weights data/chunk_weights.pt \\
        --config       configs/dp_ucvla.yaml \\
        --data_dir     data/
"""

import argparse
import os
from itertools import combinations

import torch
import yaml
from tqdm.auto import tqdm

from datasets import get_val_dataset, collate_fn
from models.clip_encoder import CLIPEncoder
from models.dp.model import UCVLADiT
from models.dp.runner import UCVLADPRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights",       type=str, required=True)
    p.add_argument("--config",        type=str, default="configs/dp_ucvla.yaml")
    p.add_argument("--data_dir",      type=str, default="data/")
    p.add_argument("--chunk_weights", type=str, default=None)
    p.add_argument("--n_rollouts",    type=int, default=1)
    p.add_argument("--val_shards",    type=str, default=None)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--max_batches",   type=int, default=None,
                   help="Cap eval batches for quick checks.")
    p.add_argument("--device",        type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    print("Loading CLIP encoder...")
    clip = CLIPEncoder().to(device)
    clip_transform = clip.get_transform()

    ckpt    = torch.load(args.weights, map_location="cpu", weights_only=False)
    n_users = ckpt.get("n_users", cfg["n_users"])
    bias_dim = ckpt.get("bias_dim", cfg["bias_dim"])

    print(f"Loading UCVLADiT (n_users={n_users}, bias_dim={bias_dim})...")
    model = UCVLADiT(
        n_users=n_users, bias_dim=bias_dim,
        hidden_size=cfg["hidden_size"], depth=cfg["depth"],
        num_heads=cfg["num_heads"], pred_horizon=cfg["pred_horizon"],
        action_dim=cfg["action_dim"], state_dim=cfg["state_dim"],
    )
    runner = UCVLADPRunner(model=model)
    model.user_bias.load_state_dict(ckpt["user_bias"])
    model.bias_proj.load_state_dict(ckpt["bias_proj"])
    runner.eval().to(device)

    # ------------------------------------------------------------------ #
    # Embedding geometry (sanity)                                          #
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        mu = model.user_bias.mu.weight.detach().cpu()  # (n_users, bias_dim)
        norms = mu.norm(dim=-1)
    print("\n── Embedding geometry ────────────────────────────────────────────")
    for i in range(n_users):
        print(f"  ||mu_{i}|| = {norms[i]:.4f}")
    for (i, j) in combinations(range(n_users), 2):
        cos = torch.nn.functional.cosine_similarity(mu[i:i+1], mu[j:j+1]).item()
        print(f"  cos(mu_{i}, mu_{j}) = {cos:+.4f}")

    # ------------------------------------------------------------------ #
    # Chunk weights (optional, for bucketing)                              #
    # ------------------------------------------------------------------ #
    chunk_weights_map: dict | None = None
    if args.chunk_weights and os.path.exists(args.chunk_weights):
        chunk_weights_map = torch.load(
            args.chunk_weights, map_location="cpu", weights_only=False
        )
        print(f"\nLoaded chunk weights ({len(chunk_weights_map):,} chunks)")
    else:
        print("\nNo chunk weights — bucketed divergence will be skipped.")

    # ------------------------------------------------------------------ #
    # Val data                                                             #
    # ------------------------------------------------------------------ #
    shards_dir = os.path.join(args.data_dir, "mug_handover_webdataset")
    val_shards = None
    if args.val_shards:
        import glob
        all_shards = sorted(glob.glob(os.path.join(shards_dir, "shard-*.tar")))
        val_shards = [all_shards[int(i)] for i in args.val_shards.split(",")]
        print(f"Val shards: {[os.path.basename(s) for s in val_shards]}")
    val_ds = get_val_dataset(shards_dir, clip_transform, cfg["state_dim"], val_shards)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0,
    )

    # ------------------------------------------------------------------ #
    # Eval loop                                                            #
    # ------------------------------------------------------------------ #
    pair_indices = list(combinations(range(n_users), 2))
    T = cfg["pred_horizon"]

    div_sum_global  = {pair: 0.0 for pair in pair_indices}      # scalar
    div_sum_bucket  = {pair: torch.zeros(3) for pair in pair_indices}
    bucket_count    = torch.zeros(3)
    n_samples       = 0

    print(f"\nRunning divergence eval ({args.n_rollouts} rollouts/sample)...")

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(val_loader, desc="eval")):
            if args.max_batches is not None and bi >= args.max_batches:
                break

            images   = batch["images"].to(device)
            states   = batch["states"].to(device, dtype=dtype)
            ep_idxs  = batch.get("episode_idxs")
            chunk_sfs = batch.get("chunk_start_frames")
            B = images.shape[0]

            clip_tokens = clip(images)

            # Generate predictions under each user bias on the same input.
            # CRITICAL: share the starting noise across users so the divergence
            # we measure is purely from the bias, not from independent rollouts.
            preds: list[torch.Tensor] = []  # each (B, T, action_dim)
            num_steps = runner.num_inference_steps
            step_size = 1.0 / num_steps
            for uid in range(n_users):
                uid_tensor = torch.full((B,), uid, device=device, dtype=torch.long)
                acc = torch.zeros(B, T, cfg["action_dim"], device=device, dtype=dtype)
                for r in range(args.n_rollouts):
                    # Same seed across users for rollout r → shared noise init
                    g = torch.Generator(device=device).manual_seed(bi * 10_000 + r)
                    x = torch.randn(
                        B, T, cfg["action_dim"],
                        device=device, dtype=dtype, generator=g,
                    )
                    for i in range(num_steps):
                        t_step = torch.full((B,), i * step_size, device=device, dtype=dtype)
                        vel = runner.model(
                            noisy_action=x, t=t_step,
                            clip_tokens=clip_tokens, state=states,
                            user_id=uid_tensor,
                        )
                        x = x + vel * step_size
                    acc += x
                preds.append(acc / args.n_rollouts)

            # Per-frame L2 divergence per pair: (B, T)
            pair_div: dict[tuple[int, int], torch.Tensor] = {}
            for (i, j) in pair_indices:
                d = (preds[i] - preds[j]).norm(dim=-1).cpu()  # (B, T)
                pair_div[(i, j)] = d
                div_sum_global[(i, j)] += d.sum().item()

            n_samples += B

            # Bucketed divergence by chunk_weight
            if chunk_weights_map is not None and ep_idxs is not None:
                import numpy as np
                w_batch = torch.ones(B, T)
                for k in range(B):
                    key = (int(ep_idxs[k]), int(chunk_sfs[k]))
                    w_arr = chunk_weights_map.get(key, None)
                    if w_arr is not None:
                        w_batch[k] = torch.from_numpy(np.asarray(w_arr))

                # bucket: low (<0.8), mid (0.8–1.2), high (>=1.2)
                bucket_idx = torch.zeros_like(w_batch, dtype=torch.long)
                bucket_idx[(w_batch >= 0.8) & (w_batch < 1.2)] = 1
                bucket_idx[w_batch >= 1.2] = 2

                for b in range(3):
                    mask = bucket_idx == b  # (B, T)
                    n_b  = mask.sum().item()
                    if n_b == 0:
                        continue
                    bucket_count[b] += n_b
                    for pair in pair_indices:
                        div_sum_bucket[pair][b] += pair_div[pair][mask].sum().item()

    # ------------------------------------------------------------------ #
    # Results                                                              #
    # ------------------------------------------------------------------ #
    n_frames = n_samples * T
    print(f"\nEvaluated on {n_samples} samples × {T} frames = {n_frames} frame slots")

    print("\n── Mean pairwise L2 divergence (per-frame, all frames) ───────────")
    for (i, j) in pair_indices:
        mean_d = div_sum_global[(i, j)] / max(n_frames, 1)
        print(f"  u{i} ↔ u{j}: {mean_d:.5f}")

    if chunk_weights_map is not None and bucket_count.sum() > 0:
        print("\n── Divergence by chunk_weight bucket ─────────────────────────────")
        labels = ["low  (<0.8)   ", "mid  (0.8-1.2)", "high (>=1.2)  "]
        header = "                " + "  ".join(f"u{i}↔u{j}" for (i, j) in pair_indices)
        print(header)
        for b, lbl in enumerate(labels):
            n_b = int(bucket_count[b].item())
            if n_b == 0:
                print(f"  {lbl}  (n=0)")
                continue
            row = "  ".join(
                f"{div_sum_bucket[pair][b].item() / n_b:.5f}"
                for pair in pair_indices
            )
            print(f"  {lbl}  (n={n_b:6d})  {row}")

        print("\n  Read: divergence should be HIGHER in 'high' bucket than 'low'.")
        print("        Flat → embeddings affect the wrong frames (or chunk_weights are off).")
        print("        Near-zero everywhere → embeddings have no behavioral effect.")


if __name__ == "__main__":
    main()
