"""
WebDataset loading for ucvla-proxy.

Each shard sample has:
  image.jpg   — (384, 768) RGB PIL image: two 384×384 frames side-by-side
                (left = wrist cam, right = env cam in RDT2 format)
  action.npy  — (24, 20) float32 action chunk
  meta.json   — {"sub_task_instruction_key": ..., "user_id": int}

We crop the left 384×384 half (wrist view) and apply the CLIP preprocess
transform before returning.  State is not stored in the webdataset — a
zeros tensor of shape (1, state_dim) is returned.
"""

import glob
import os
from typing import Callable, Optional

import numpy as np
import torch
import webdataset as wds
from PIL import Image


def _no_split(src):
    yield from src


class SampleMapper:
    """Top-level callable so it can be pickled by multiprocessing spawn."""

    def __init__(
        self,
        clip_transform: Callable,
        state_dim: int,
        chunk_weights: Optional[dict] = None,
    ) -> None:
        self.clip_transform = clip_transform
        self.state_dim = state_dim
        self.chunk_weights = chunk_weights  # {(episode_idx, chunk_start_frame): (T,) ndarray}

    def __call__(self, sample: dict) -> dict:
        img: Image.Image = sample["image.jpg"]
        w = img.width // 2
        img = img.crop((0, 0, w, img.height))
        img_tensor = self.clip_transform(img)

        action = torch.from_numpy(sample["action.npy"].copy())  # (T, 10)
        meta: dict = sample["meta.json"]
        user_id           = int(meta.get("user_id", -1))
        episode_idx       = int(meta.get("episode_idx", -1))
        chunk_start_frame = int(meta.get("chunk_start_frame", -1))
        state = torch.zeros(1, self.state_dim, dtype=torch.float32)

        out = {
            "image":             img_tensor,
            "action":            action,
            "state":             state,
            "user_id":           user_id,
            "episode_idx":       episode_idx,
            "chunk_start_frame": chunk_start_frame,
        }

        if self.chunk_weights is not None:
            w_arr = self.chunk_weights.get((episode_idx, chunk_start_frame), None)
            if w_arr is None:
                w_arr = np.ones(action.shape[0], dtype=np.float32)
            out["chunk_weights"] = torch.from_numpy(w_arr)

        return out


def get_train_dataset(
    shards_dir: str,
    clip_transform: Callable,
    state_dim: int,
    chunk_weights_path: Optional[str] = None,
) -> wds.WebDataset:
    shards = sorted(glob.glob(os.path.join(shards_dir, "shard-*.tar")))
    assert shards, f"No shards found in {shards_dir}"

    chunk_weights = None
    if chunk_weights_path:
        chunk_weights = torch.load(chunk_weights_path, map_location="cpu", weights_only=False)

    dataset = (
        wds.WebDataset(shards, shardshuffle=True, nodesplitter=_no_split, resampled=True)
        .shuffle(2048, initial=2048)
        .decode("pil")
        .map(SampleMapper(clip_transform, state_dim, chunk_weights))
    )
    return dataset


def get_val_dataset(
    shards_dir: str,
    clip_transform: Callable,
    state_dim: int,
    val_shards: Optional[list[str]] = None,
) -> wds.WebDataset:
    all_shards = sorted(glob.glob(os.path.join(shards_dir, "shard-*.tar")))
    assert all_shards, f"No shards found in {shards_dir}"
    if val_shards:
        shards = val_shards
    else:
        # Pick one shard per user from shard_index.json (scales to any number of users)
        index_path = os.path.join(shards_dir, "shard_index.json")
        if os.path.exists(index_path):
            import json
            with open(index_path) as f:
                shard_index = json.load(f)  # {user_id_str: [shard_filename, ...]}
            seen, shards = set(), []
            for uid_str, uid_shards in sorted(shard_index.items()):
                for s in uid_shards:
                    full = os.path.join(shards_dir, s)
                    if full not in seen:
                        shards.append(full)
                        seen.add(full)
                        break
        else:
            shards = [all_shards[-1]]

    dataset = (
        wds.WebDataset(shards, shardshuffle=False, nodesplitter=_no_split,
                       workersplitter=_no_split)
        .decode("pil")
        .map(SampleMapper(clip_transform, state_dim))
    )
    return dataset


def collate_fn(examples: list[dict]) -> dict[str, torch.Tensor]:
    out = {
        "images":             torch.stack([e["image"] for e in examples]),
        "actions":            torch.stack([e["action"] for e in examples]),
        "states":             torch.stack([e["state"] for e in examples]),
        "user_ids":           torch.tensor([e["user_id"] for e in examples], dtype=torch.long),
        "episode_idxs":       torch.tensor([e["episode_idx"] for e in examples], dtype=torch.long),
        "chunk_start_frames": torch.tensor([e["chunk_start_frame"] for e in examples], dtype=torch.long),
    }
    if "chunk_weights" in examples[0]:
        out["chunk_weights"] = torch.stack([e["chunk_weights"] for e in examples])
    return out
