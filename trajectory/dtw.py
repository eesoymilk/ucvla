import numpy as np


def dtw_path(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
    """Classic DTW with backtracking.

    Args:
        a: (T_a, D)
        b: (T_b, D)

    Returns:
        Alignment path as list of (a_idx, b_idx) from (0, 0) to (T_a-1, T_b-1).
    """
    T_a, T_b = len(a), len(b)
    cost = np.sum((a[:, None] - b[None]) ** 2, axis=-1)  # (T_a, T_b)

    dp = np.full((T_a, T_b), np.inf)
    dp[0, 0] = cost[0, 0]
    for i in range(1, T_a):
        dp[i, 0] = dp[i - 1, 0] + cost[i, 0]
    for j in range(1, T_b):
        dp[0, j] = dp[0, j - 1] + cost[0, j]
    for i in range(1, T_a):
        for j in range(1, T_b):
            dp[i, j] = cost[i, j] + min(dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1])

    i, j = T_a - 1, T_b - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            move = np.argmin([dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1]])
            if move == 0:
                i -= 1; j -= 1
            elif move == 1:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    path.reverse()
    return path


def dba(trajs: list[np.ndarray], n_iters: int = 10) -> np.ndarray:
    """DTW Barycenter Averaging over variable-length trajectories.

    Initializes the center as the sequence closest to the median length, then
    iteratively aligns all sequences via DTW and re-averages.

    Args:
        trajs:   List of (L_i, D) float32 arrays.
        n_iters: Refinement iterations.

    Returns:
        (L_center, D) barycenter.
    """
    lengths = np.array([len(t) for t in trajs])
    init_idx = int(np.argmin(np.abs(lengths - np.median(lengths))))
    center = trajs[init_idx].copy().astype(np.float64)

    for it in range(n_iters):
        assoc: list[list[np.ndarray]] = [[] for _ in range(len(center))]
        for traj in trajs:
            for ci, ti in dtw_path(center.astype(np.float32), traj):
                assoc[ci].append(traj[ti])
        center = np.array([
            np.mean(frames, axis=0) if frames else center[t]
            for t, frames in enumerate(assoc)
        ])
        print(f"  DBA iter {it + 1}/{n_iters}")

    return center.astype(np.float32)
