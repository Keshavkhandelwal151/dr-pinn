"""Compare two sets of vessel centrelines (e.g. extracted vs ground truth, or
U-Net-mask graph vs manual-mask graph in M5). Topology-free: works even when
one set splits or merges segments differently."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree


def densify(path: ArrayLike, spacing: float = 0.5) -> NDArray[np.float64]:
    p = np.asarray(path, dtype=np.float64)
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        t = np.linspace(0, 1, n + 1)[1:, None]
        out.append(a + t * (b - a))
    return np.vstack(out)


@dataclass(frozen=True)
class CentrelineComparison:
    recall: float          # share of reference centreline within tol of the test centreline
    precision: float       # share of test centreline within tol of the reference centreline
    radius_rel_error: NDArray[np.float64]  # per matched test segment: r_test / r_ref - 1
    matched: int
    unmatched: int
    radius_test: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))  # matched pairs, test side
    radius_ref: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))   # matched pairs, reference side

    @property
    def radius_median_abs_error(self) -> float:
        return float(np.median(np.abs(self.radius_rel_error))) if self.radius_rel_error.size else float("nan")


def compare_centrelines(test_paths: Sequence[ArrayLike], test_radii: ArrayLike,
                        ref_paths: Sequence[ArrayLike], ref_radii: ArrayLike,
                        tol_px: float = 2.0, end_trim: float = 0.2,
                        min_majority: float = 0.8, mask: ArrayLike | None = None) -> CentrelineComparison:
    """Centreline recall/precision and per-segment radius agreement.

    A test segment is matched to the reference segment that is nearest for at
    least `min_majority` of its points (ignoring `end_trim` of its length at
    each end, where junction geometry dominates). `mask` optionally restricts
    which test segments are scored (e.g. only reliable widths).
    """
    t_dense = [densify(p) for p in test_paths]
    r_dense = [densify(p) for p in ref_paths]
    t_all, r_all = np.vstack(t_dense), np.vstack(r_dense)
    r_ids = np.concatenate([np.full(len(p), i) for i, p in enumerate(r_dense)])
    tree_t, tree_r = cKDTree(t_all), cKDTree(r_all)
    recall = float(np.mean(tree_t.query(r_all)[0] <= tol_px))
    precision = float(np.mean(tree_r.query(t_all)[0] <= tol_px))

    test_radii = np.asarray(test_radii, float)
    ref_radii = np.asarray(ref_radii, float)
    use = np.ones(len(t_dense), bool) if mask is None else np.asarray(mask, bool)
    errs, unmatched, pairs = [], 0, []
    for i, pts in enumerate(t_dense):
        if not use[i]:
            continue
        n = len(pts)
        core = pts[int(end_trim * n): max(int((1 - end_trim) * n), int(end_trim * n) + 1)]
        dist, idx = tree_r.query(core)
        ids = r_ids[idx[dist <= tol_px]]
        if ids.size < 0.5 * len(core):
            unmatched += 1
            continue
        best = np.bincount(ids).argmax()
        if np.mean(ids == best) < min_majority:
            unmatched += 1
            continue
        errs.append(test_radii[i] / ref_radii[best] - 1.0)
        pairs.append((test_radii[i], ref_radii[best]))
    pr = np.array(pairs, float).reshape(-1, 2)
    return CentrelineComparison(recall, precision, np.array(errs), len(errs), unmatched, pr[:, 0], pr[:, 1])
