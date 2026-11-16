"""Segmentation and artery/vein metrics, computed inside the field of view."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata
from skimage.morphology import skeletonize


def roc_auc(scores: NDArray, labels: NDArray[np.bool_]) -> float:
    """Area under the ROC curve (Mann-Whitney form, ties averaged)."""
    s = np.asarray(scores, np.float64).ravel()
    y = np.asarray(labels, bool).ravel()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def segmentation_metrics(prob: NDArray, ref: NDArray[np.bool_], fov: NDArray[np.bool_],
                         threshold: float = 0.5) -> dict[str, float]:
    """Dice, IoU, sensitivity, specificity, accuracy and AUC over FOV pixels."""
    fov = np.asarray(fov, bool)
    p = np.asarray(prob, np.float64)[fov]
    t = np.asarray(ref, bool)[fov]
    pred = p >= threshold
    tp, fp = int((pred & t).sum()), int((pred & ~t).sum())
    fn, tn = int((~pred & t).sum()), int((~pred & ~t).sum())

    def div(a, b):
        return float(a / b) if b else float("nan")

    return {"dice": div(2 * tp, 2 * tp + fp + fn), "iou": div(tp, tp + fp + fn),
            "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp), "precision": div(tp, tp + fp),
            "accuracy": div(tp + tn, tp + tn + fp + fn), "auc": roc_auc(p, t)}


def av_metrics(p_artery: NDArray, p_vein: NDArray, av_labels: NDArray[np.uint8],
               ref_vessel: NDArray[np.bool_]) -> dict[str, float]:
    """Artery/vein classification on the reference vessel CENTRELINE.

    Scored only where the reference says artery (1) or vein (2); overlap (3)
    and uncertain (4) pixels are skipped. A pixel is called artery when
    p_artery > p_vein. Sensitivity = arteries found, specificity = veins found.
    """
    centre = skeletonize(np.asarray(ref_vessel, bool))
    lab = np.asarray(av_labels)
    art, vein = centre & (lab == 1), centre & (lab == 2)
    call_art = np.asarray(p_artery) > np.asarray(p_vein)
    tp, fn = int((call_art & art).sum()), int((~call_art & art).sum())
    tn, fp = int((~call_art & vein).sum()), int((call_art & vein).sum())
    n = tp + fn + tn + fp
    return {"av_accuracy": (tp + tn) / n if n else float("nan"),
            "av_sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
            "av_specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "av_centreline_pixels": n}


def av_label_metrics(pred_labels: NDArray, av_labels: NDArray[np.uint8], ref_vessel: NDArray[np.bool_],
                     prefix: str = "av") -> dict[str, float]:
    """Artery/vein scores of a label map (0 none, 1 artery, 2 vein, 3 both) on the reference centreline.

    Scored where the reference says artery (1) or vein (2). A pixel left
    unlabelled by the prediction counts as wrong (it is missing from both
    trees); a pixel labelled both (a crossing) counts as artery for artery
    pixels and vein for vein pixels.
    """
    centre = skeletonize(np.asarray(ref_vessel, bool))
    lab, p = np.asarray(av_labels), np.asarray(pred_labels)
    art, vein = centre & (lab == 1), centre & (lab == 2)
    is_a, is_v = (p == 1) | (p == 3), (p == 2) | (p == 3)
    ok = (art & is_a) | (vein & is_v)
    n = int(art.sum() + vein.sum())
    tp_a, fp_a, fn_a = int((art & is_a).sum()), int((vein & is_a & ~is_v).sum()), int((art & ~is_a).sum())
    tp_v, fp_v, fn_v = int((vein & is_v).sum()), int((art & is_v & ~is_a).sum()), int((vein & ~is_v).sum())

    def div(a, b):
        return float(a / b) if b else float("nan")

    return {f"{prefix}_accuracy": div(int(ok.sum()), n),
            f"{prefix}_artery_precision": div(tp_a, tp_a + fp_a), f"{prefix}_artery_recall": div(tp_a, tp_a + fn_a),
            f"{prefix}_artery_f1": div(2 * tp_a, 2 * tp_a + fp_a + fn_a),
            f"{prefix}_vein_precision": div(tp_v, tp_v + fp_v), f"{prefix}_vein_recall": div(tp_v, tp_v + fn_v),
            f"{prefix}_vein_f1": div(2 * tp_v, 2 * tp_v + fp_v + fn_v),
            f"{prefix}_unlabelled": div(int(((art | vein) & ~is_a & ~is_v).sum()), n)}
