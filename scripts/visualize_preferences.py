#!/usr/bin/env python3
"""
Render per-user preference-highlighted videos.

Loads DBA prototypes (or computes them) then renders one MP4 per episode
with a green→red border and timeline bar indicating the preference weight.

Usage:
    cd ~/Codes/ucvla
    uv run --with "opencv-python" python scripts/visualize_preferences.py \\
        --zarr_dir umi_data \\
        --out_dir  outputs/preference_viz \\
        --user_id  0 \\
        --n_episodes 3 \\
        --rdt2_dir ~/Codes/RDT2
"""

import argparse
import os
import sys

import numpy as np


def render_video(
    frames: np.ndarray,
    weights: np.ndarray,
    out_path: str,
    fps: int = 15,
) -> None:
    import cv2

    L, H, W, _ = frames.shape
    scale = 2
    fw, fh = W * scale, H * scale
    bar_h = 50

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh + bar_h))

    w_max = weights.max() if weights.max() > 0 else 1.0
    curve_pts = np.array([
        [int(t / L * fw), int((1 - weights[t] / w_max) * (bar_h - 8)) + 4]
        for t in range(L)
    ], dtype=np.int32)

    for t in range(L):
        img = cv2.resize(
            cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR),
            (fw, fh), interpolation=cv2.INTER_NEAREST,
        )

        w_norm = float(np.clip(weights[t] / w_max, 0, 1))
        border = max(3, int(w_norm * 14))
        hue = int((1 - w_norm) * 60)
        color = cv2.cvtColor(np.uint8([[[hue, 220, 230]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        img[:border] = color;  img[-border:] = color
        img[:, :border] = color;  img[:, -border:] = color

        if weights[t] > 2.0:
            cv2.putText(img, "PREFERENCE", (fw // 2 - 100, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 60, 255), 2)
        cv2.putText(img, f"t={t:03d}  w={weights[t]:.2f}", (8, fh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        bar = np.full((bar_h, fw, 3), 30, dtype=np.uint8)
        cv2.polylines(bar, [curve_pts], isClosed=False, color=(120, 120, 120), thickness=1)
        for tt in range(t + 1):
            x = curve_pts[tt, 0]
            ww = float(np.clip(weights[tt] / w_max, 0, 1))
            col = cv2.cvtColor(np.uint8([[[int((1-ww)*60), 180, 200]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            cv2.line(bar, (x, curve_pts[tt, 1]), (x, bar_h - 4), col, 1)
        cur_x = curve_pts[t, 0]
        cv2.line(bar, (cur_x, 0), (cur_x, bar_h), (255, 255, 255), 2)
        cv2.circle(bar, (cur_x, curve_pts[t, 1]), 4, (255, 255, 255), -1)

        writer.write(np.vstack([img, bar]))

    writer.release()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zarr_dir", required=True)
    p.add_argument("--rdt2_dir", default=os.path.expanduser("~/Codes/RDT2"))
    p.add_argument("--out_dir", default="outputs/preference_viz")
    p.add_argument("--user_id", type=int, default=None, help="0/1/2 or omit for all")
    p.add_argument("--n_episodes", type=int, default=3)
    p.add_argument("--dba_iters", type=int, default=10)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--load_prototypes", type=str, default=None)
    args = p.parse_args()

    sys.path.insert(0, args.rdt2_dir)
    from data.umi.codecs.imagecodecs_numcodecs import register_codecs
    register_codecs()

    from zarr_loader import load_episodes
    from trajectory import dba, compute_weights, global_norm_stats, make_traj

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading episodes...")
    all_by_user: dict[int, list] = {}
    for uid in range(3):
        path = os.path.join(args.zarr_dir, f"replay_buffer_u{uid}.zarr")
        all_by_user[uid] = load_episodes(path, load_rgb=True)
        lens = [len(ep["pos"]) for ep in all_by_user[uid]]
        print(f"  u{uid}: {len(lens)} episodes, len=[{min(lens)},{max(lens)}]")

    all_flat = [ep for eps in all_by_user.values() for ep in eps]
    pos_mean, pos_std, grip_std = global_norm_stats(all_flat)

    prototypes: dict[int, np.ndarray] = {}
    proto_path = os.path.join(args.out_dir, "prototypes.npz")
    if args.load_prototypes or os.path.exists(proto_path):
        npz = np.load(args.load_prototypes or proto_path)
        for uid in range(3):
            prototypes[uid] = npz[f"u{uid}"]
        print("Loaded cached prototypes.")
    else:
        print("Computing DBA prototypes...")
        for uid in range(3):
            other_trajs = [
                make_traj(ep, pos_mean, pos_std, grip_std)
                for u, eps in all_by_user.items() if u != uid
                for ep in eps
            ]
            print(f"  u{uid}: {len(other_trajs)} episodes...")
            prototypes[uid] = dba(other_trajs, n_iters=args.dba_iters)
        np.savez(proto_path, **{f"u{uid}": prototypes[uid] for uid in range(3)})
        print(f"Saved prototypes → {proto_path}")

    users = [args.user_id] if args.user_id is not None else [0, 1, 2]
    for uid in users:
        for ep in all_by_user[uid][: args.n_episodes]:
            traj = make_traj(ep, pos_mean, pos_std, grip_std)
            weights = compute_weights(traj, prototypes[uid])
            out = os.path.join(args.out_dir, f"u{uid}_ep{ep['idx']:03d}.mp4")
            print(f"  u{uid} ep{ep['idx']:03d}: max_w={weights.max():.2f} → {out}")
            render_video(ep["rgb"], weights, out, fps=args.fps)


if __name__ == "__main__":
    main()
