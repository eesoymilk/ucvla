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

    def __init__(self, clip_transform: Callable, state_dim: int) -> None:
        self.clip_transform = clip_transform
        self.state_dim = state_dim

    def __call__(self, sample: dict) -> dict:
        # Crop wrist view (left half of the side-by-side image)
        img: Image.Image = sample["image.jpg"]
        w = img.width // 2
        img = img.crop((0, 0, w, img.height))
        img_tensor = self.clip_transform(img)         # (3, 224, 224)

        action = torch.from_numpy(sample["action.npy"].copy())  # (T, 10)
        meta: dict = sample["meta.json"]
        user_id = int(meta.get("user_id", -1))
        state = torch.zeros(1, self.state_dim, dtype=torch.float32)

        return {
            "image": img_tensor,
            "action": action,
            "state": state,
            "user_id": user_id,
        }


def get_train_dataset(
    shards_dir: str,
    clip_transform: Callable,
    state_dim: int,
) -> wds.WebDataset:
    shards = sorted(glob.glob(os.path.join(shards_dir, "shard-*.tar")))
    assert shards, f"No shards found in {shards_dir}"

    dataset = (
        wds.WebDataset(
            shards,
            shardshuffle=True,
            nodesplitter=_no_split,
            resampled=True,
        )
        .shuffle(2048, initial=2048)
        .decode("pil")
        .map(SampleMapper(clip_transform, state_dim))
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
    shards = val_shards if val_shards else [all_shards[-1]]

    dataset = (
        wds.WebDataset(shards, shardshuffle=False, nodesplitter=_no_split,
                       workersplitter=_no_split)
        .decode("pil")
        .map(SampleMapper(clip_transform, state_dim))
    )
    return dataset


def collate_fn(examples: list[dict]) -> dict[str, torch.Tensor]:
    images = torch.stack([e["image"] for e in examples])       # (B, 3, 224, 224)
    actions = torch.stack([e["action"] for e in examples])     # (B, T, 20)
    states = torch.stack([e["state"] for e in examples])       # (B, 1, state_dim)
    user_ids = torch.tensor([e["user_id"] for e in examples], dtype=torch.long)  # (B,)
    return {"images": images, "actions": actions, "states": states, "user_ids": user_ids}
