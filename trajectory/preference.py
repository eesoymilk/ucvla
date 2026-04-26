import numpy as np
from scipy.ndimage import uniform_filter1d

from trajectory.dtw import dtw_path


def global_norm_stats(all_episodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Global mean/std of the 10D episode-relative action across all episodes."""
    all_action = np.concatenate([ep["action"] for ep in all_episodes])
    return all_action.mean(axis=0), all_action.std(axis=0).clip(min=1e-6)


def make_traj(ep: dict, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize (L, 10) episode action for DTW."""
    return ((ep["action"] - mean) / std).astype(np.float32)


def compute_weights(
    ep_traj: np.ndarray,
    prototype: np.ndarray,
    smooth_window: int = 9,
) -> np.ndarray:
    """Per-frame preference weight via DTW deviation from cross-user prototype.

    Each frame's weight = L2 distance to its DTW-aligned prototype frame,
    smoothed to suppress sensor noise, normalized to mean = 1.

    Args:
        ep_traj:      (L, D) normalized episode trajectory.
        prototype:    (L_proto, D) DBA cross-mean of other users.
        smooth_window: Uniform smoothing window in frames.

    Returns:
        (L,) weight array, mean = 1.
    """
    path = dtw_path(ep_traj, prototype)

    ep_to_proto: dict[int, list[int]] = {}
    for ep_i, proto_j in path:
        ep_to_proto.setdefault(ep_i, []).append(proto_j)

    weights = np.zeros(len(ep_traj), dtype=np.float32)
    for ep_i, proto_js in ep_to_proto.items():
        proto_avg = prototype[proto_js].mean(axis=0)
        weights[ep_i] = float(np.linalg.norm(ep_traj[ep_i] - proto_avg))

    if smooth_window > 1:
        weights = uniform_filter1d(weights, size=smooth_window).astype(np.float32)

    mean_w = weights.mean()
    return weights / mean_w if mean_w > 0 else weights
