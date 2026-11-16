"""One eye: vessel/artery/vein probabilities -> graphs -> Monte Carlo risk -> patch table.

No lesion information is used to build the vessel geometry. (The PDF suggests
masking lesions out of the vessel map first; that would cut gaps - and new
vessel ends with their own boundary conditions - exactly at lesions, leaking
the outcome into the physics.) Instead, the share of lesion pixels the model
labelled as vessel is reported as "contamination".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..segmentation.av import av_graphs, masks_from_probabilities
from ..uncertainty.montecarlo import MCConfig, tree_risk
from ..vessel_graph import geometry_features
from .patches import EyeGeometry, estimate_fovea, patch_table


@dataclass(frozen=True)
class M7Config:
    fov_diameter_px: int = 540     # working scale = scale of the DRIVE photos the M5 model learned from
    patch_px: int = 32             # ~1/17 of the field of view
    threshold: float = 0.5
    mc: MCConfig = field(default_factory=MCConfig)
    # how artery/vein maps become the two trees (segmentation.av.av_label_map). "threshold" is the
    # pre-registered rule; "vote" (one decision per vessel segment) is the M12 fix for patchy A/V labels.
    av_method: str = "threshold"

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.av_method == "threshold":  # unchanged settings for runs made before av_method existed
            d.pop("av_method")
        return d


def disc_from_mask(od_mask: NDArray[np.bool_]) -> tuple[tuple[float, float], float]:
    """Centre (row, col) and equal-area diameter (px) of the largest optic-disc region."""
    m = np.asarray(od_mask, bool)
    lab, n = ndimage.label(m)
    if n == 0:
        raise ValueError("optic-disc mask is empty")
    big = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    r, c = ndimage.center_of_mass(big)
    return (float(r), float(c)), float(2 * np.sqrt(big.sum() / np.pi))


def _tree_geometry(graph, disc_rc, disc_px, mc: MCConfig) -> dict | None:
    if graph is None or graph.n_edges == 0:
        return None
    px_um = float(np.mean(mc.disc_um_range)) / disc_px
    net = graph.to_network(px_um)
    _, comp = net.connected_components()
    dist = np.linalg.norm(graph.node_rc - np.asarray(disc_rc), axis=1)
    inlets = [int(np.flatnonzero(comp == c)[np.argmin(dist[comp == c])]) for c in np.unique(comp)]
    f = geometry_features(net, inlets, edge_paths=graph.paths_xy_m(px_um))
    return {"tortuosity": f["tortuosity"], "radius_um": f["radius_um"], "murray_deviation": f["murray_deviation"]}


def analyse_eye(image_id: str, prob: NDArray, fov: NDArray[np.bool_], od_mask: NDArray[np.bool_],
                lesion_masks: dict[str, NDArray] | None, cfg: M7Config = M7Config()) -> tuple[list[dict], dict]:
    """Patch rows + per-eye summary. `prob` is (3, H, W): vessel, artery, vein at working scale."""
    fov = np.asarray(fov, bool)
    disc_rc, disc_px = disc_from_mask(od_mask)
    eye = EyeGeometry(disc_rc, disc_px, estimate_fovea(disc_rc, disc_px, fov), True)
    vessel, artery, vein = masks_from_probabilities(prob, cfg.threshold, cfg.av_method)
    vessel &= fov
    graphs = av_graphs(vessel, artery & fov, vein & fov, fov=fov, vessel_prob=prob[0])
    trees, geometry, info = {}, {}, {"image_id": image_id, "disc_diameter_px": disc_px}
    for name, g in (("artery", graphs.artery), ("vein", graphs.vein)):
        risk = tree_risk(g, disc_rc, disc_px, name, cfg.mc)
        trees[name] = (g, risk)
        geometry[name] = _tree_geometry(g, disc_rc, disc_px, cfg.mc) if risk is not None else None
        info[f"{name}_segments"] = 0 if g is None else g.n_edges
        info[f"{name}_segments_solved"] = 0 if risk is None else int(risk.solved.sum())
        info[f"{name}_rank_stability"] = float("nan") if risk is None else risk.rank_stability
    info.update({f"graph_{k}": v for k, v in graphs.summary().items()})
    if lesion_masks:
        red = np.zeros_like(fov)
        for k in ("MA", "HE"):
            if k in lesion_masks:
                red |= np.asarray(lesion_masks[k], bool)
        info["red_lesion_pixels"] = int(red.sum())
        info["red_lesion_labelled_vessel"] = float((red & vessel).sum() / red.sum()) if red.any() else float("nan")
    rows = patch_table(image_id, fov.shape, fov, cfg.patch_px, eye, trees, vessel, lesion_masks, geometry)
    info["patches_analysed"] = len(rows)
    return rows, info
