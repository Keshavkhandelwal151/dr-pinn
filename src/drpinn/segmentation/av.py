"""Separate arterial and venous vessel graphs (fixes assumption A3, docs/PLAN.md).

Arteries and veins cross; in a single vessel mask each crossing becomes a
false junction that joins the two trees. With multi-label artery/vein maps,
each tree is extracted on its own; crossing pixels belong to both maps, so
both trees stay continuous and neither sees the other's crossings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..vessel_graph.extract import ExtractionConfig, VesselGraph, extract_vessel_graph


@dataclass(frozen=True, eq=False)
class AVGraphs:
    combined: VesselGraph
    artery: VesselGraph | None
    vein: VesselGraph | None

    def summary(self) -> dict[str, float]:
        out = {}
        for name, g in (("combined", self.combined), ("artery", self.artery), ("vein", self.vein)):
            if g is None:
                continue
            out[f"{name}_segments"] = g.n_edges
            out[f"{name}_components"] = int(g.to_network(1.0).connected_components()[0])
            out[f"{name}_crossings_split"] = g.stats["crossings_resolved"]
            out[f"{name}_possible_crossings"] = g.stats["possible_crossings"]
        return out


def _graph(mask: NDArray[np.bool_], prob: NDArray | None, fov, config) -> VesselGraph | None:
    if mask.sum() < (config or ExtractionConfig()).min_component_area_px:
        return None
    soft = None
    if prob is not None:
        near = ndimage.binary_dilation(mask, iterations=2)
        soft = np.where(near, np.clip(prob, 0, 1), 0.0)
    try:
        return extract_vessel_graph(mask, fov=fov, probability=soft, config=config)
    except ValueError:  # nothing left after cleaning
        return None


def av_graphs(vessel_mask: NDArray[np.bool_], artery_mask: NDArray[np.bool_], vein_mask: NDArray[np.bool_],
              fov: NDArray[np.bool_] | None = None, vessel_prob: NDArray | None = None,
              config: ExtractionConfig | None = None) -> AVGraphs:
    """Combined, arterial and venous graphs. Masks must be boolean and the same size."""
    v = np.asarray(vessel_mask, bool)
    a = np.asarray(artery_mask, bool) & v
    w = np.asarray(vein_mask, bool) & v
    combined = extract_vessel_graph(v, fov=fov, probability=vessel_prob, config=config)
    return AVGraphs(combined, _graph(a, vessel_prob, fov, config), _graph(w, vessel_prob, fov, config))


AV_METHODS = ("threshold", "argmax", "vote")


def _neighbour_count(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    k = np.ones((3, 3), int)
    k[1, 1] = 0
    return ndimage.convolve(mask.astype(int), k, mode="constant")


def vessel_pieces(vessel: NDArray[np.bool_]) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
    """Split a vessel mask into pieces between junctions.

    Returns (piece label of every vessel pixel (0 outside), junction-cluster
    label of every pixel (0 outside), number of pieces). The skeleton is cut
    at every branch point or crossing; each vessel pixel then belongs to the
    nearest remaining centreline piece.
    """
    from skimage.morphology import skeletonize

    v = np.asarray(vessel, bool)
    pieces = np.zeros(v.shape, np.int64)
    if not v.any():
        return pieces, pieces.copy(), 0
    skel = skeletonize(v)
    junction = skel & (_neighbour_count(skel) >= 3)
    junction_zone = ndimage.binary_dilation(junction, np.ones((3, 3), bool))
    seeds, n = ndimage.label(skel & ~junction_zone, np.ones((3, 3), int))
    clusters, _ = ndimage.label(junction_zone & skel, np.ones((3, 3), int))
    if n == 0:  # a skeleton made only of junction pixels: one piece
        pieces[v] = 1
        return pieces, clusters, 1
    _, (ri, ci) = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    pieces[v] = seeds[ri[v], ci[v]]
    return pieces, clusters, n


def av_label_map(prob: NDArray, threshold: float = 0.5, method: str = "vote",
                 vessel: NDArray[np.bool_] | None = None) -> NDArray[np.uint8]:
    """Label map 0 background, 1 artery, 2 vein, 3 both (crossing) from a (3, H, W) probability map.

    Methods:
      threshold  each of artery / vein independently >= threshold (the original
                 rule). Vessel pixels where neither head is confident are left
                 unlabelled, which breaks the trees into fragments.
      argmax     every vessel pixel goes to the larger of p_artery and p_vein.
      vote       (default) one decision per vessel piece between junctions:
                 the piece is artery when its mean p_artery exceeds its mean
                 p_vein, weighted by vessel probability. Pixel noise inside a
                 vessel can no longer flip parts of it. At a junction where an
                 artery piece and a vein piece meet (a crossing), the junction
                 pixels are given to both trees so each stays continuous.
    """
    if method not in AV_METHODS:
        raise ValueError(f"method must be one of {AV_METHODS}")
    prob = np.asarray(prob, np.float64)
    v = prob[0] >= threshold if vessel is None else np.asarray(vessel, bool)
    pa, pv = prob[1], prob[2]
    out = np.zeros(v.shape, np.uint8)
    if method == "threshold":
        a, w = v & (pa >= threshold), v & (pv >= threshold)
        out[a & ~w], out[w & ~a], out[a & w] = 1, 2, 3
        return out
    if method == "argmax":
        out[v & (pa >= pv)] = 1
        out[v & (pa < pv)] = 2
        return out

    pieces, clusters, n = vessel_pieces(v)
    if n == 0:
        return out
    weight = np.clip(prob[0], 1e-3, 1.0)
    lab = pieces[v]
    wsum = np.bincount(lab, weight[v], n + 1)
    diff = np.bincount(lab, weight[v] * (pa[v] - pv[v]), n + 1) / np.maximum(wsum, 1e-12)
    piece_class = np.where(diff >= 0, 1, 2).astype(np.uint8)
    piece_class[0] = 0
    out[v] = piece_class[lab]

    # crossings: a junction touched by both an artery piece and a vein piece belongs to both trees
    if clusters.max() > 0:
        radius = ndimage.distance_transform_edt(v)
        n_cl = int(clusters.max())
        reach = ndimage.maximum(radius, clusters, index=np.arange(1, n_cl + 1))
        h, w = v.shape
        k3 = np.ones((3, 3), bool)
        for c, box in enumerate(ndimage.find_objects(clusters), start=1):
            if box is None:
                continue
            r = int(np.ceil(reach[c - 1])) + 1
            pad = r + 2
            sl = (slice(max(box[0].start - pad, 0), min(box[0].stop + pad, h)),
                  slice(max(box[1].start - pad, 0), min(box[1].stop + pad, w)))
            vv = v[sl]
            zone = ndimage.binary_dilation(clusters[sl] == c, k3, iterations=r) & vv
            ring = ndimage.binary_dilation(zone, k3) & vv
            classes = set(piece_class[np.unique(pieces[sl][ring])].tolist()) - {0}
            if {1, 2} <= classes:  # an artery piece and a vein piece meet here: a crossing
                out[sl][zone] = 3
    return out


def masks_from_probabilities(prob: NDArray, threshold: float = 0.5,
                             method: str = "threshold") -> tuple[NDArray, NDArray, NDArray]:
    """(vessel, artery, vein) boolean masks from a (3, H, W) probability map.

    `method` is passed to `av_label_map`. The default stays "threshold" so the
    pre-registered M7 analysis is reproduced exactly; use "vote" for new runs
    (see scripts/m12_evaluate.py for the comparison).
    """
    vessel = prob[0] >= threshold
    if method == "threshold":
        return vessel, vessel & (prob[1] >= threshold), vessel & (prob[2] >= threshold)
    lab = av_label_map(prob, threshold, method, vessel)
    return vessel, (lab == 1) | (lab == 3), (lab == 2) | (lab == 3)
