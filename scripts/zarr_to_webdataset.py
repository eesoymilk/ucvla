#!/usr/bin/env python3
"""Convert a UMI zarr replay buffer to WebDataset format (single-arm, 10-D actions).

Each output sample is one action chunk:
  {idx}.image.jpg   — (384, 768, 3) uint8: wrist cam duplicated side-by-side
  {idx}.action.npy  — (24, 10) float32: right-arm EEF pose + gripper
  {idx}.meta.json   — {"sub_task_instruction_key", "task_id", "user_id",
                        "episode_idx", "chunk_start_frame"}

Action layout (10-D, right arm only):
  dims 0-2  : EEF position relative to chunk-start frame (metres)
  dims 3-8  : EEF rotation in 6-D continuous rep (first two columns of rot matrix)
  dim  9    : gripper width (raw metres)

Usage:
  cd ~/Codes/ucvla
  uv run --with "zarr<3,imagecodecs[all],webdataset,pillow,scipy" \\
      python scripts/zarr_to_webdataset.py \\
      --zarr_path  umi_data/replay_buffer_u0.zarr \\
      --out_dir    data/mug_handover_webdataset \\
      --rdt2_dir   ~/Codes/RDT2
"""

import argparse
import io
import json
import os
import sys
import tarfile

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

import zarr

# ── constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE   = 24
IMAGE_OUT    = 384
JPEG_QUALITY = 95


# ── helpers ───────────────────────────────────────────────────────────────────

def build_pose_mats(pos: np.ndarray, rot_aa: np.ndarray) -> np.ndarray:
    """(T, 3), (T, 3) → (T, 4, 4) homogeneous pose matrices."""
    T = len(pos)
    mats = np.zeros((T, 4, 4), dtype=np.float64)
    mats[:, :3, :3] = Rotation.from_rotvec(rot_aa).as_matrix()
    mats[:, :3, 3] = pos
    mats[:, 3, 3] = 1.0
    return mats


def mat_to_pose9d(mat: np.ndarray) -> np.ndarray:
    """(T, 4, 4) → (T, 9): [pos(3), rot_col0(3), rot_col1(3)]."""
    return np.concatenate([mat[:, :3, 3], mat[:, :3, 0], mat[:, :3, 1]], axis=-1)


def build_action_chunk(
    pos: np.ndarray,
    rot_aa: np.ndarray,
    gripper: np.ndarray,
    start: int,
) -> np.ndarray:
    """(T_ep, 3/3/1) → (CHUNK_SIZE, 10) float32 relative action chunk."""
    T_ep = len(pos)
    end  = min(start + CHUNK_SIZE, T_ep)
    pad  = CHUNK_SIZE - (end - start)

    chunk_pos  = pos[start:end]
    chunk_rot  = rot_aa[start:end]
    chunk_grip = gripper[start:end]

    if pad > 0:
        chunk_pos  = np.concatenate([chunk_pos,  np.tile(chunk_pos[-1:],  (pad, 1))])
        chunk_rot  = np.concatenate([chunk_rot,  np.tile(chunk_rot[-1:],  (pad, 1))])
        chunk_grip = np.concatenate([chunk_grip, np.tile(chunk_grip[-1:], (pad, 1))])

    pose_mats = build_pose_mats(chunk_pos, chunk_rot)   # (24, 4, 4)
    rel_mats  = np.linalg.inv(pose_mats[0])[None] @ pose_mats  # relative to frame 0

    pose9d = mat_to_pose9d(rel_mats).astype(np.float32)         # (24, 9)
    return np.concatenate([pose9d, chunk_grip.astype(np.float32)], axis=-1)  # (24, 10)


def encode_image(frame: np.ndarray) -> bytes:
    img = Image.fromarray(frame).resize((IMAGE_OUT, IMAGE_OUT), Image.LANCZOS)
    arr = np.asarray(img)
    tiled = np.concatenate([arr, arr], axis=1)  # (384, 768, 3)
    buf = io.BytesIO()
    Image.fromarray(tiled).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def add_to_tar(tar: tarfile.TarFile, key: str, ext: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=f"{key}.{ext}")
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


# ── main ──────────────────────────────────────────────────────────────────────

def convert(
    zarr_path: str,
    out_dir: str,
    task_key: str,
    instruction: str,
    stride: int,
    shard_size: int,
    skip_short: bool,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "instructions.json"), "w") as f:
        json.dump({task_key: instruction}, f, indent=2)

    root = zarr.open(zarr.ZipStore(zarr_path, mode="r"))
    ep_ends   = root["meta"]["episode_ends"][:]
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    n_eps     = len(ep_ends)

    sample_idx = shard_idx = n_skipped = 0
    tar = tarfile.open(os.path.join(out_dir, f"shard-{shard_idx:06d}.tar"), "w")

    for ep_idx in range(n_eps):
        ep_s, ep_e = int(ep_starts[ep_idx]), int(ep_ends[ep_idx])
        ep_len = ep_e - ep_s

        if ep_len < CHUNK_SIZE:
            if skip_short:
                n_skipped += 1
                continue

        ep_rgb  = root["data"]["camera0_rgb"][ep_s:ep_e]
        ep_pos  = root["data"]["robot0_eef_pos"][ep_s:ep_e]
        ep_rot  = root["data"]["robot0_eef_rot_axis_angle"][ep_s:ep_e]
        ep_grip = root["data"]["robot0_gripper_width"][ep_s:ep_e]
        task_id = int(root["data"]["task_id"][ep_s])
        user_id = int(root["data"]["user_id"][ep_s])

        for cs in range(0, ep_len, stride):
            if sample_idx > 0 and sample_idx % shard_size == 0:
                tar.close()
                shard_idx += 1
                tar = tarfile.open(os.path.join(out_dir, f"shard-{shard_idx:06d}.tar"), "w")

            key    = f"{sample_idx:08d}"
            action = build_action_chunk(ep_pos, ep_rot, ep_grip, cs)
            buf    = io.BytesIO(); np.save(buf, action)
            meta   = {
                "sub_task_instruction_key": task_key,
                "task_id": task_id,
                "user_id": user_id,
                "episode_idx": ep_idx,
                "chunk_start_frame": ep_s + cs,
            }
            add_to_tar(tar, key, "image.jpg", encode_image(ep_rgb[cs]))
            add_to_tar(tar, key, "action.npy", buf.getvalue())
            add_to_tar(tar, key, "meta.json", json.dumps(meta).encode())
            sample_idx += 1

        if (ep_idx + 1) % 10 == 0 or ep_idx == n_eps - 1:
            print(f"  {ep_idx + 1}/{n_eps} episodes  {sample_idx} samples")

    tar.close()
    print(f"\nDone: {n_eps - n_skipped} episodes, {sample_idx} samples, "
          f"{shard_idx + 1} shards → {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zarr_path",   required=True)
    p.add_argument("--out_dir",     required=True)
    p.add_argument("--rdt2_dir",    default=os.path.expanduser("~/Codes/RDT2"),
                   help="RDT2 repo root for image codec registration")
    p.add_argument("--task_key",    default="mug_handover")
    p.add_argument("--instruction", default="Pick up the mug and hand it to the user.")
    p.add_argument("--stride",      type=int, default=1)
    p.add_argument("--shard_size",  type=int, default=1000)
    p.add_argument("--skip_short",  action="store_true")
    args = p.parse_args()

    sys.path.insert(0, args.rdt2_dir)
    from data.umi.codecs.imagecodecs_numcodecs import register_codecs
    register_codecs()

    convert(
        zarr_path=args.zarr_path,
        out_dir=args.out_dir,
        task_key=args.task_key,
        instruction=args.instruction,
        stride=args.stride,
        shard_size=args.shard_size,
        skip_short=args.skip_short,
    )


if __name__ == "__main__":
    main()
