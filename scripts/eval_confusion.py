#!/usr/bin/env python3
"""
Preference-aware cross-user confusion matrix for UCVLA proxy.

Only evaluates on high-preference chunks (where the user's behavior is
discriminative, identified via DBA deviation weights). Within those chunks,
uses weighted MSE so even borderline frames contribute proportionally.

Usage:
    PYTHONPATH="$(pwd)" uv run python scripts/eval_confusion.py \\
        --weights      outputs/dp_ucvla/stage1_10d/step-50000/ucvla_weights.pt \\
        --chunk_weights data/chunk_weights.pt \\
        --config        configs/dp_ucvla.yaml \\
        --data_dir      data/
"""

import argparse
import os
from collections import defaultdict

import torch
import yaml

from datasets import get_val_dataset, collate_fn
from models.clip_encoder import CLIPEncoder
from models.dp.model import UCVLADiT
from models.dp.runner import UCVLADPRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--config",  type=str, default="configs/dp_ucvla.yaml")
    p.add_argument("--data_dir", type=str, default="data/")
    p.add_argument("--chunk_weights", type=str, default=None,
                   help="Path to chunk_weights.pt. If omitted falls back to uniform MSE.")
    p.add_argument("--pref_threshold", type=float, default=1.2,
                   help="Min mean chunk weight to be considered a preference chunk.")
    p.add_argument("--n_rollouts", type=int, default=5,
                   help="ODE rollouts per sample to reduce variance.")
    p.add_argument("--val_shards", type=str, default=None,
                   help="Comma-separated shard indices to use for val, e.g. '5,35,65'. "
                        "Default: last shard only (may miss users).")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
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
    # Chunk weights                                                         #
    # ------------------------------------------------------------------ #
    chunk_weights_map: dict | None = None
    if args.chunk_weights and os.path.exists(args.chunk_weights):
        chunk_weights_map = torch.load(
            args.chunk_weights, map_location="cpu", weights_only=False
        )
        print(f"Loaded chunk weights ({len(chunk_weights_map):,} chunks)")
        print(f"Preference threshold: mean weight > {args.pref_threshold}")
    else:
        print("No chunk weights — using uniform MSE on all chunks (fallback)")

    # ------------------------------------------------------------------ #
    # Val data (with chunk_weights so meta.json is available)              #
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
        val_ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2,
    )

    # ------------------------------------------------------------------ #
    # Eval loop                                                            #
    # ------------------------------------------------------------------ #
    # errors[(uid_true, uid_pred)] = [weighted_mse, ...]
    errors: dict[tuple[int, int], list[float]] = defaultdict(list)
    n_pref_chunks = 0
    n_skipped     = 0

    print(f"\nRunning preference-aware eval ({args.n_rollouts} rollouts/chunk)...")

    with torch.no_grad():
        for batch in val_loader:
            images   = batch["images"].to(device)
            actions  = batch["actions"].to(device, dtype=dtype)
            states   = batch["states"].to(device, dtype=dtype)
            uid_true = batch["user_ids"].to(device)
            ep_idxs  = batch.get("episode_idxs")
            chunk_sfs = batch.get("chunk_start_frames")

            clip_tokens = clip(images)  # (B, 196, 512)

            # Build per-sample weight vectors and preference mask
            B, T, _ = actions.shape
            w_batch  = torch.ones(B, T, device=device, dtype=dtype)  # default uniform
            pref_mask = torch.ones(B, dtype=torch.bool)              # default: all chunks

            if chunk_weights_map is not None and ep_idxs is not None:
                for i in range(B):
                    key = (int(ep_idxs[i]), int(chunk_sfs[i]))
                    w_arr = chunk_weights_map.get(key, None)
                    if w_arr is not None:
                        w_batch[i] = torch.from_numpy(w_arr).to(device)
                        pref_mask[i] = w_batch[i].mean() >= args.pref_threshold
                    else:
                        pref_mask[i] = False

            # Skip non-preference chunks
            if chunk_weights_map is not None:
                n_pref  = pref_mask.sum().item()
                n_skipped += (B - n_pref)
                if n_pref == 0:
                    continue
                images      = images[pref_mask]
                actions     = actions[pref_mask]
                states      = states[pref_mask]
                uid_true    = uid_true[pref_mask]
                clip_tokens = clip_tokens[pref_mask]
                w_batch     = w_batch[pref_mask]
                n_pref_chunks += n_pref
            else:
                n_pref_chunks += B

            # Average N rollouts to reduce ODE variance
            for uid_pred in range(n_users):
                uid_tensor = torch.full_like(uid_true, uid_pred)
                pred_accum = torch.zeros_like(actions)
                for _ in range(args.n_rollouts):
                    pred_accum += runner.predict_action(
                        clip_tokens=clip_tokens,
                        state=states,
                        user_id=uid_tensor,
                    )
                pred = pred_accum / args.n_rollouts

                # Weighted MSE per sample: (w * (pred-gt)²).mean(dim=[T, action_dim])
                w = w_batch.unsqueeze(-1)  # (B, T, 1)
                err = (w * (pred - actions).pow(2)).mean(dim=[1, 2])  # (B,)
                for i, ut in enumerate(uid_true.tolist()):
                    errors[(ut, uid_pred)].append(err[i].item())

    # ------------------------------------------------------------------ #
    # Results                                                              #
    # ------------------------------------------------------------------ #
    print(f"\nEvaluated on {n_pref_chunks} preference chunks "
          f"(skipped {n_skipped} non-preference chunks)")

    print("\n── Preference-weighted confusion matrix (lower=better) ───────────")
    print("   rows=true user, cols=predicted bias\n")
    print("        " + "  ".join(f"bias_{j}" for j in range(n_users)))
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
        others  = [
            sum(errors[(ut, up)]) / len(errors[(ut, up)])
            for up in range(n_users) if up != ut and errors[(ut, up)]
        ]
        if others and correct < min(others):
            wins += 1
            print(f"  user_{ut}: correct bias WINS ✓  ({correct:.5f} < {min(others):.5f})")
        else:
            best_other = min(others) if others else float("nan")
            print(f"  user_{ut}: correct bias LOSES ✗  ({correct:.5f} vs min_other={best_other:.5f})")

    print(f"\n  {wins}/{n_users} users have correct bias winning")
    if wins == n_users:
        print("  PASS → port to RDT2")
    else:
        print("  FAIL → check cosim trends or enable preference weighting in training")


if __name__ == "__main__":
    main()
