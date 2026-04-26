#!/usr/bin/env python3
"""
Compute per-chunk preference weights from DBA prototypes + zarr episodes.

For each episode, DTW-aligns the 10D trajectory to its user's cross-mean
prototype and extracts per-frame deviation weights. These are sliced into
24-frame chunks matching the WebDataset layout.

Output: data/chunk_weights.pt
  dict mapping (episode_idx, chunk_start_frame) → (24,) float32 weight array

Usage:
    cd ~/Codes/ucvla
    uv run --with "zarr<3,imagecodecs[all],scipy" \\
        python scripts/compute_chunk_weights.py \\
        --zarr_path umi_data/combined_replay_buffer.zarr \\
        --proto_npz outputs/preference_viz/prototypes.npz \\
        --out_path  data/chunk_weights.pt \\
        --rdt2_dir  ~/Codes/RDT2
"""

import argparse
import os
import sys

import numpy as np
import torch
import zarr
from scipy.spatial.transform import Rotation

CHUNK_SIZE = 24


def episode_action_10d(pos, rot_aa, gripper):
    L = len(pos)
    mats = np.zeros((L, 4, 4), dtype=np.float64)
    mats[:, :3, :3] = Rotation.from_rotvec(rot_aa).as_matrix()
    mats[:, :3, 3] = pos
    mats[:, 3, 3] = 1.0
    base_inv = np.linalg.inv(mats[0])
    rel = base_inv[None] @ mats
    pose9d = np.concatenate([rel[:, :3, 3], rel[:, :3, 0], rel[:, :3, 1]], axis=-1)
    return np.concatenate([pose9d, gripper], axis=-1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr_path", default="umi_data/combined_replay_buffer.zarr")
    p.add_argument("--proto_npz", default="outputs/preference_viz/prototypes.npz")
    p.add_argument("--out_path",  default="data/chunk_weights.pt")
    p.add_argument("--rdt2_dir",  default=os.path.expanduser("~/Codes/RDT2"))
    p.add_argument("--stride",    type=int, default=1)
    args = p.parse_args()

    sys.path.insert(0, args.rdt2_dir)
    from data.umi.codecs.imagecodecs_numcodecs import register_codecs
    register_codecs()

    sys.path.insert(0, ".")
    from trajectory.preference import compute_weights

    # Load prototypes
    npz = np.load(args.proto_npz)
    prototypes = {int(k[1:]): npz[k] for k in npz}
    print(f"Prototypes: { {uid: p.shape for uid, p in prototypes.items()} }")

    # Load zarr
    root = zarr.open(zarr.ZipStore(args.zarr_path, mode="r"))
    ep_ends   = root["meta"]["episode_ends"][:]
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    n_eps     = len(ep_ends)

    # Compute global norm stats from all episodes
    print("Computing normalisation stats...")
    all_actions = []
    for s, e in zip(ep_starts, ep_ends):
        pos    = root["data"]["robot0_eef_pos"][s:e]
        rot_aa = root["data"]["robot0_eef_rot_axis_angle"][s:e]
        grip   = root["data"]["robot0_gripper_width"][s:e]
        all_actions.append(episode_action_10d(pos, rot_aa, grip))
    all_actions = np.concatenate(all_actions)
    action_mean = all_actions.mean(axis=0)
    action_std  = all_actions.std(axis=0).clip(min=1e-6)
    print(f"  action shape: {all_actions.shape}")

    # Compute per-chunk weights
    print("Computing chunk weights...")
    weights_dict: dict[tuple[int, int], np.ndarray] = {}

    for ep_idx in range(n_eps):
        s, e   = int(ep_starts[ep_idx]), int(ep_ends[ep_idx])
        ep_len = e - s
        uid    = int(root["data"]["user_id"][s])

        pos    = root["data"]["robot0_eef_pos"][s:e]
        rot_aa = root["data"]["robot0_eef_rot_axis_angle"][s:e]
        grip   = root["data"]["robot0_gripper_width"][s:e]

        action = episode_action_10d(pos, rot_aa, grip)
        traj   = (action - action_mean) / action_std
        w      = compute_weights(traj, prototypes[uid])  # (L,)

        for cs in range(0, ep_len, args.stride):
            chunk_w = w[cs : cs + CHUNK_SIZE]
            if len(chunk_w) < CHUNK_SIZE:
                chunk_w = np.pad(chunk_w, (0, CHUNK_SIZE - len(chunk_w)), mode="edge")
            weights_dict[(ep_idx, s + cs)] = chunk_w.astype(np.float32)

        if (ep_idx + 1) % 50 == 0 or ep_idx == n_eps - 1:
            print(f"  {ep_idx + 1}/{n_eps} episodes")

    print(f"Total chunks: {len(weights_dict)}")
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    torch.save(weights_dict, args.out_path)
    print(f"Saved → {args.out_path}")


if __name__ == "__main__":
    main()
