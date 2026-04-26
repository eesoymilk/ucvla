"""Load UMI zarr replay buffers into episode dicts."""

import numpy as np
import zarr
from scipy.spatial.transform import Rotation


def _build_pose_mats(pos: np.ndarray, rot_aa: np.ndarray) -> np.ndarray:
    """(L, 3), (L, 3) → (L, 4, 4) homogeneous pose matrices."""
    L = len(pos)
    mats = np.zeros((L, 4, 4), dtype=np.float64)
    mats[:, :3, :3] = Rotation.from_rotvec(rot_aa).as_matrix()
    mats[:, :3, 3] = pos
    mats[:, 3, 3] = 1.0
    return mats


def _episode_action_10d(
    pos: np.ndarray,      # (L, 3) absolute EEF position
    rot_aa: np.ndarray,   # (L, 3) axis-angle
    gripper: np.ndarray,  # (L, 1) gripper width
) -> np.ndarray:
    """Convert full episode to 10D episode-relative action (same space as training).

    All frames are expressed relative to episode frame 0, matching the
    coordinate system used in the WebDataset (except chunks reset at each
    chunk start; here we reset only once at episode start).

    Returns (L, 10): [rel_pos(3), rot_col0(3), rot_col1(3), gripper(1)]
    """
    pose_mats = _build_pose_mats(pos, rot_aa)        # (L, 4, 4)
    base_inv  = np.linalg.inv(pose_mats[0])          # (4, 4)
    rel_mats  = base_inv[None] @ pose_mats            # (L, 4, 4)

    rel_pos  = rel_mats[:, :3, 3]                    # (L, 3)
    rot_col0 = rel_mats[:, :3, 0]                    # (L, 3)
    rot_col1 = rel_mats[:, :3, 1]                    # (L, 3)
    pose9d   = np.concatenate([rel_pos, rot_col0, rot_col1], axis=-1).astype(np.float32)
    return np.concatenate([pose9d, gripper.astype(np.float32)], axis=-1)  # (L, 10)


def load_episodes(zarr_path: str, load_rgb: bool = True) -> list[dict]:
    """Load all episodes from a zarr ZipStore replay buffer.

    Each episode dict has:
        idx:     episode index
        action:  (L, 10) float32  episode-relative 10D action (same space as training)
        rgb:     (L, 224, 224, 3) uint8  (only if load_rgb=True)

    Args:
        zarr_path: Path to the .zarr ZipStore file.
        load_rgb:  Whether to load camera frames (slow; skip for weight-only runs).
    """
    root = zarr.open(zarr.ZipStore(zarr_path, mode="r"))
    ep_ends   = root["meta"]["episode_ends"][:]
    ep_starts = np.concatenate([[0], ep_ends[:-1]])

    episodes = []
    for i, (s, e) in enumerate(zip(ep_starts, ep_ends)):
        pos     = root["data"]["robot0_eef_pos"][s:e]
        rot_aa  = root["data"]["robot0_eef_rot_axis_angle"][s:e]
        gripper = root["data"]["robot0_gripper_width"][s:e]

        ep: dict = {
            "idx":    i,
            "action": _episode_action_10d(pos, rot_aa, gripper),  # (L, 10)
        }
        if load_rgb:
            ep["rgb"] = root["data"]["camera0_rgb"][s:e]
        episodes.append(ep)

    return episodes
