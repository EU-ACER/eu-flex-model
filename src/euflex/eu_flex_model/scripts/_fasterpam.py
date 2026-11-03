"""
Vectorized NumPy implementation of the FasterPAM k-medoids algorithm.

Reference: Schubert & Rousseeuw (2021), "Fast and Eager k-Medoids Clustering:
O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms",
Information Systems 101. https://arxiv.org/abs/2008.05171
"""

from collections import namedtuple

import numpy as np

KMedoidsResult = namedtuple("KMedoidsResult", ["labels", "medoids"])


def _initial_assignment(
    D: np.ndarray,
    medoids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign each point to its nearest and second-nearest medoid.

    Parameters
    ----------
    D: (N, N) precomputed distance matrix
    medoids: (k,) current medoid indices (point indices into D)

    Returns
    -------
    near_rank: (N,) index into medoids[] of nearest medoid (0..k-1)
    near_dist: (N,) distance to nearest medoid
    seco_rank: (N,) index into medoids[] of second-nearest medoid; -1 if k=1
    seco_dist: (N,) distance to second-nearest medoid; inf if k=1
    """
    k = len(medoids)
    # D_med: (N, k) distances from every point to every medoid
    D_med = D[:, medoids]

    if k == 1:
        near_rank = np.zeros(len(D), dtype=np.intp)
        near_dist = D_med[:, 0].copy()
        seco_rank = np.full(len(D), -1, dtype=np.intp)
        seco_dist = np.full(len(D), np.inf)
        return near_rank, near_dist, seco_rank, seco_dist

    # argsort each row to get ranks; we only need positions 0 and 1
    order = np.argpartition(D_med, 1, axis=1)
    near_rank = order[:, 0]
    seco_rank = order[:, 1]

    # Ensure near_rank is truly the minimum (argpartition only guarantees position 0
    # is <= position 1, not that 0 is strictly the smallest of the two)
    swap_mask = (
        D_med[np.arange(len(D)), near_rank] > D_med[np.arange(len(D)), seco_rank]
    )
    near_rank[swap_mask], seco_rank[swap_mask] = (
        seco_rank[swap_mask].copy(),
        near_rank[swap_mask].copy(),
    )

    rows = np.arange(len(D))
    near_dist = D_med[rows, near_rank]
    seco_dist = D_med[rows, seco_rank]

    return near_rank, near_dist, seco_rank, seco_dist


def _update_removal_loss(
    near_rank: np.ndarray,
    near_dist: np.ndarray,
    seco_dist: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Precompute the cost delta of removing each medoid.

    removal_loss[m] = sum over points o assigned to m of (seco_dist[o] - near_dist[o])

    This is the increase in total distance if medoid m were removed and each of
    its points fell back to its second-nearest medoid.

    Parameters
    ----------
    near_rank: (N,) medoid-rank of nearest medoid per point
    near_dist: (N,) distance to nearest medoid per point
    seco_dist: (N,) distance to second-nearest medoid per point
    k: number of medoids

    Returns
    -------
    removal_loss: (k,) cost delta per medoid
    """
    delta = seco_dist - near_dist
    removal_loss = np.zeros(k)
    np.add.at(removal_loss, near_rank, delta)
    return removal_loss


def _find_best_swap(
    D: np.ndarray,
    j: int,
    near_rank: np.ndarray,
    near_dist: np.ndarray,
    seco_rank: np.ndarray,
    seco_dist: np.ndarray,
    removal_loss: np.ndarray,
) -> tuple[int, float]:
    """
    Evaluate candidate j as a new medoid; find the best medoid to evict.

    For each existing medoid b, compute the net change in total distance if we
    swap b out and replace it with j. Uses the FasterPAM O(N) accumulator trick.

    Parameters
    ----------
    D: (N, N) distance matrix
    j: candidate point index
    near_rank, near_dist, seco_rank, seco_dist: assignment arrays (N,)
    removal_loss: (k,) precomputed removal costs

    Returns
    -------
    best_b: medoid-rank of best medoid to evict; -1 if no improvement
    best_delta: cost change (negative = improvement)
    """
    N = len(near_rank)
    doj = D[np.arange(N), j]  # distances from all points to candidate j

    ploss = removal_loss.copy()
    acc = 0.0

    # Exclude j itself from the vectorized scan - handled separately below.
    # D[j,j]=0 so doj[j]=0, which is always < near_dist[j] (j is not a medoid here).
    points = np.arange(N)
    mask_not_j = points != j

    doj_all = doj[mask_not_j]
    nd_all = near_dist[mask_not_j]
    sd_all = seco_dist[mask_not_j]
    nr_all = near_rank[mask_not_j]

    # Case 1: j is closer than the point's current nearest medoid
    # -> j steals the point; net: (doj - near_dist) added to acc,
    #    and the eviction cost for near_rank[o] is reduced by (seco_dist - near_dist)
    mask1 = doj_all < nd_all
    acc += float(np.sum(doj_all[mask1] - nd_all[mask1]))
    np.add.at(ploss, nr_all[mask1], nd_all[mask1] - sd_all[mask1])

    # Case 2: j is between nearest and second-nearest
    # -> if near_rank[o]'s medoid is evicted, j limits the fallback cost
    mask2 = (~mask1) & (doj_all < sd_all)
    np.add.at(ploss, nr_all[mask2], doj_all[mask2] - sd_all[mask2])

    # Point j itself: D[j,j]=0, so j is its own nearest once it becomes a medoid.
    # Its contribution: distance drops from near_dist[j] to 0 -> acc += -near_dist[j].
    # Its fallback removal_loss[near_rank[j]] no longer applies for j.
    nr_j = int(near_rank[j])
    nd_j = float(near_dist[j])
    sd_j = float(seco_dist[j])
    acc += -nd_j
    # Reverse j's removal_loss contribution for its current medoid
    ploss[nr_j] -= sd_j - nd_j

    total = ploss + acc
    best_b = int(np.argmin(total))
    best_delta = float(total[best_b])

    if best_delta < 0.0:
        return best_b, best_delta
    return -1, best_delta


def _do_swap(
    D: np.ndarray,
    medoids: np.ndarray,
    near_rank: np.ndarray,
    near_dist: np.ndarray,
    seco_rank: np.ndarray,
    seco_dist: np.ndarray,
    b: int,
    j: int,
) -> None:
    """
    Apply swap: replace medoid at rank b with point j. Updates arrays in-place.

    Three cases per point o after medoid b is replaced by j:
    - Case A: near_rank[o] != b  -> check if j is closer than near or seco
    - Case B: near_rank[o] == b and D[o,j] <= near_dist[o]  -> j becomes new nearest
    - Case C: near_rank[o] == b and D[o,j] > near_dist[o]   -> must rescan all medoids

    Parameters
    ----------
    D: (N, N) distance matrix
    medoids: (k,) medoid indices, modified in-place
    near_rank, near_dist, seco_rank, seco_dist: assignment arrays, modified in-place
    b: medoid-rank being evicted
    j: point index becoming new medoid
    """
    N = len(near_rank)
    medoids[b] = j

    doj = D[np.arange(N), j]

    assigned_to_b = near_rank == b

    # --- Case A: point not assigned to b ---
    not_b = ~assigned_to_b

    # Case A sub-case: j is closer than current nearest
    caseA1 = not_b & (doj < near_dist)
    seco_rank[caseA1] = near_rank[caseA1]
    seco_dist[caseA1] = near_dist[caseA1]
    near_rank[caseA1] = b  # b now points to j (medoids[b] = j)
    near_dist[caseA1] = doj[caseA1]

    # Case A sub-case: j is between nearest and second-nearest
    caseA2 = not_b & (~caseA1) & (doj < seco_dist)
    seco_rank[caseA2] = b
    seco_dist[caseA2] = doj[caseA2]

    # --- Case B: assigned to b, j is at least as close ---
    # seco_rank/seco_dist are unchanged: they already point to the closest non-b
    # medoid, and no medoid at rank != b was modified by this swap.
    caseB = assigned_to_b & (doj <= near_dist)
    near_rank[caseB] = b
    near_dist[caseB] = doj[caseB]

    # --- Case C: assigned to b, j is farther -> full rescan ---
    caseC = assigned_to_b & (~caseB)
    if np.any(caseC):
        c_indices = np.nonzero(caseC)[0]
        D_med = D[np.ix_(c_indices, medoids)]
        order = np.argpartition(D_med, 1, axis=1)
        nr = order[:, 0]
        sr = order[:, 1]
        swap = (
            D_med[np.arange(len(c_indices)), nr] > D_med[np.arange(len(c_indices)), sr]
        )
        nr[swap], sr[swap] = sr[swap].copy(), nr[swap].copy()
        near_rank[c_indices] = nr
        near_dist[c_indices] = D_med[np.arange(len(c_indices)), nr]
        seco_rank[c_indices] = sr
        seco_dist[c_indices] = D_med[np.arange(len(c_indices)), sr]

    # Point j itself: nearest is self (distance 0)
    near_rank[j] = b
    near_dist[j] = 0.0
    # seco for j: nearest medoid that is not j itself
    D_j = D[j, medoids].copy()
    D_j[b] = np.inf
    if len(medoids) > 1:
        sr_j = int(np.argmin(D_j))
        seco_rank[j] = sr_j
        seco_dist[j] = D_j[sr_j]
    else:
        seco_rank[j] = -1
        seco_dist[j] = np.inf


def fasterpam(
    D: np.ndarray,
    n_clusters: int,
    random_state: int,
    max_iter: int = 100,
) -> KMedoidsResult:
    """
    Run FasterPAM k-medoids on a precomputed distance matrix.

    Parameters
    ----------
    D: (N, N) symmetric precomputed distance matrix (float64)
    n_clusters: number of clusters k
    random_state: integer seed for reproducibility
    max_iter: maximum number of outer iterations

    Returns
    -------
    KMedoidsResult with fields:
        labels: (N,) cluster assignment per point (0..k-1)
        medoids: (k,) point indices of medoids
    """
    N = len(D)
    k = n_clusters
    rng = np.random.default_rng(random_state)

    medoids = rng.choice(N, size=k, replace=False)
    near_rank, near_dist, seco_rank, seco_dist = _initial_assignment(D, medoids)

    if k == 1:
        # Only one medoid: pick the point minimising sum of distances to all others.
        best_m = int(np.argmin(D.sum(axis=1)))
        labels = np.zeros(N, dtype=np.intp)
        return KMedoidsResult(labels=labels, medoids=np.array([best_m]))

    for _ in range(max_iter):
        n_swaps = 0
        removal_loss = _update_removal_loss(near_rank, near_dist, seco_dist, k)
        medoid_set = set(medoids.tolist())

        for j in range(N):
            if j in medoid_set:
                continue
            best_b, best_delta = _find_best_swap(
                D, j, near_rank, near_dist, seco_rank, seco_dist, removal_loss
            )
            if best_b == -1:
                continue
            old_medoid = int(medoids[best_b])
            _do_swap(D, medoids, near_rank, near_dist, seco_rank, seco_dist, best_b, j)
            medoid_set.discard(old_medoid)
            medoid_set.add(j)
            removal_loss = _update_removal_loss(near_rank, near_dist, seco_dist, k)
            n_swaps += 1

        if n_swaps == 0:
            break

    labels = near_rank.copy()
    return KMedoidsResult(labels=labels, medoids=medoids.copy())
