"""Standard classification and regression metrics with heat-map figures (numpy + matplotlib only).

Used by scripts/m12_evaluate.py for every model in the project:
  * vessel segmentation      - binary, per pixel inside the field of view
  * artery / vein            - binary (artery vs vein) on labelled vessel pixels
  * DR grade triage          - 5 classes (ICDR 0-4) and referable (grade >= 2)

Which metric means what
-----------------------
accuracy    share of correct calls. Misleading when one class dominates
            (about 88 % of a fundus photo is background, so "all background"
            already scores 0.88): read it together with precision and recall.
precision   of everything called positive, the share that really is positive.
recall      of everything really positive, the share that was found (= sensitivity).
f1          harmonic mean of precision and recall (= Dice for segmentation).
r2          coefficient of determination, 1 - SS_res / SS_tot. A REGRESSION
            metric: it is reported only for continuous targets - vessel radius
            (predicted vs expert) and expected DR grade (sum_g g p_g vs true
            grade). R2 on 0/1 class labels would not be meaningful.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def confusion(y_true: ArrayLike, y_pred: ArrayLike, n_classes: int) -> NDArray[np.int64]:
    """Rows = true class, columns = predicted class."""
    t = np.asarray(y_true, np.int64).ravel()
    p = np.asarray(y_pred, np.int64).ravel()
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if t.size and (min(t.min(), p.min()) < 0 or max(t.max(), p.max()) >= n_classes):
        raise ValueError(f"labels must be in [0, {n_classes})")
    return np.bincount(t * n_classes + p, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def report_from_confusion(cm: NDArray, names: Sequence[str]) -> dict:
    """Accuracy, per-class precision / recall / specificity / F1, macro and weighted averages."""
    cm = np.asarray(cm, np.int64)
    total = int(cm.sum())
    per = {}
    for k, name in enumerate(names):
        tp = int(cm[k, k])
        fn = int(cm[k].sum()) - tp
        fp = int(cm[:, k].sum()) - tp
        tn = total - tp - fn - fp
        per[name] = {"support": tp + fn, "precision": _div(tp, tp + fp), "recall": _div(tp, tp + fn),
                     "specificity": _div(tn, tn + fp), "f1": _div(2 * tp, 2 * tp + fp + fn)}
    present = [n for n in names if per[n]["support"] > 0]
    out = {"n": total, "accuracy": _div(np.trace(cm), total), "per_class": per, "confusion_matrix": cm.tolist(),
           "class_names": list(names)}
    for key in ("precision", "recall", "f1"):
        vals = np.array([per[n][key] for n in present], float)
        sup = np.array([per[n]["support"] for n in present], float)
        ok = np.isfinite(vals)
        out[f"macro_{key}"] = float(vals[ok].mean()) if ok.any() else float("nan")
        out[f"weighted_{key}"] = float((vals[ok] * sup[ok]).sum() / sup[ok].sum()) if ok.any() and sup[ok].sum() else float("nan")
    return out


def classification_report(y_true: ArrayLike, y_pred: ArrayLike, names: Sequence[str]) -> dict:
    return report_from_confusion(confusion(y_true, y_pred, len(names)), names)


def binary_report(y_true: ArrayLike, y_pred: ArrayLike, scores: ArrayLike | None = None) -> dict:
    """Accuracy, precision, recall, specificity, F1 (= Dice), IoU and, with scores, ROC AUC."""
    t = np.asarray(y_true, bool).ravel()
    p = np.asarray(y_pred, bool).ravel()
    tp, fp = int((t & p).sum()), int((~t & p).sum())
    fn, tn = int((t & ~p).sum()), int((~t & ~p).sum())
    out = {"n": t.size, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "accuracy": _div(tp + tn, t.size), "precision": _div(tp, tp + fp), "recall": _div(tp, tp + fn),
           "specificity": _div(tn, tn + fp), "f1": _div(2 * tp, 2 * tp + fp + fn), "iou": _div(tp, tp + fp + fn),
           "balanced_accuracy": float(np.nanmean([_div(tp, tp + fn), _div(tn, tn + fp)]))
           if (tp + fn) or (tn + fp) else float("nan")}
    if scores is not None:
        from .segmentation.metrics import roc_auc
        out["auc"] = roc_auc(np.asarray(scores, float).ravel(), t)
    return out


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """1 - SS_res / SS_tot. NaN with fewer than 2 values or a constant target. Can be negative."""
    t = np.asarray(y_true, float).ravel()
    p = np.asarray(y_pred, float).ravel()
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    if t.size < 2:
        return float("nan")
    ss_tot = float(((t - t.mean()) ** 2).sum())
    return float("nan") if ss_tot == 0 else float(1 - ((t - p) ** 2).sum() / ss_tot)


def regression_report(y_true: ArrayLike, y_pred: ArrayLike) -> dict:
    t = np.asarray(y_true, float).ravel()
    p = np.asarray(y_pred, float).ravel()
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    if not t.size:
        return {"n": 0, "r2": float("nan"), "mae": float("nan"), "rmse": float("nan"), "pearson_r": float("nan")}
    r = float(np.corrcoef(t, p)[0, 1]) if t.size > 1 and t.std() > 0 and p.std() > 0 else float("nan")
    return {"n": int(t.size), "r2": r2_score(t, p), "mae": float(np.abs(t - p).mean()),
            "rmse": float(np.sqrt(((t - p) ** 2).mean())), "pearson_r": r}


# ---------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_confusion_heatmap(cm: ArrayLike, names: Sequence[str], path: str | Path, title: str = "") -> Path:
    """Two panels: counts, and row-normalised share (= recall on the diagonal)."""
    plt = _plt()
    cm = np.asarray(cm, float)
    rows = cm.sum(1, keepdims=True)
    share = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    size = max(3.2, 0.75 * len(names) + 1.6)
    fig, axes = plt.subplots(1, 2, figsize=(2 * size + 1.0, size))
    for ax, data, sub, fmt in ((axes[0], cm, "counts", "{:,.0f}"), (axes[1], share, "share of each true class", "{:.2f}")):
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max() if data.max() > 0 else 1)
        ax.set_xticks(range(len(names)), names, rotation=45 if len(max(names, key=len)) > 4 else 0, ha="right"
                      if len(max(names, key=len)) > 4 else "center")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(sub, fontsize=10)
        hi = data.max() / 2 if data.max() > 0 else 0.5
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if data[i, j] > hi else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_metric_heatmap(rows: Sequence[dict], metrics: Sequence[str], path: str | Path, row_key: str = "image_id",
                        title: str = "") -> Path:
    """Images (rows) x metrics (columns), coloured 0-1: shows at a glance which photos and metrics are weak."""
    plt = _plt()
    data = np.array([[float(r.get(m, np.nan)) for m in metrics] for r in rows], float)
    labels = [str(r.get(row_key, i)) for i, r in enumerate(rows)]
    fig, ax = plt.subplots(figsize=(1.1 * len(metrics) + 2.5, 0.32 * len(rows) + 1.8))
    im = ax.imshow(np.ma.masked_invalid(data), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics)), metrics, rotation=40, ha="right")
    ax.set_yticks(range(len(rows)), labels, fontsize=7)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_error_maps(rgb: NDArray, panels: Sequence[tuple[NDArray, str]], path: str | Path, title: str = "") -> Path:
    """Photo plus colour-coded spatial error maps (where on the retina the model is wrong)."""
    plt = _plt()
    items = [(rgb, "photo")] + list(panels)
    fig, axes = plt.subplots(1, len(items), figsize=(3.6 * len(items), 3.9))
    for ax, (im, t) in zip(np.atleast_1d(axes), items, strict=True):
        ax.imshow(im)
        ax.set_title(t, fontsize=8)
        ax.set_axis_off()
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def vessel_error_rgb(pred: NDArray[np.bool_], ref: NDArray[np.bool_], fov: NDArray[np.bool_]) -> NDArray[np.float64]:
    """White = hit, red = extra (false positive), blue = missed (false negative)."""
    pred, ref, fov = (np.asarray(a, bool) for a in (pred, ref, fov))
    out = np.zeros((*pred.shape, 3))
    out[pred & ref & fov] = (1, 1, 1)
    out[pred & ~ref & fov] = (1, 0.15, 0.15)
    out[~pred & ref & fov] = (0.25, 0.55, 1)
    return out


def av_error_rgb(pred_labels: NDArray, ref_labels: NDArray) -> NDArray[np.float64]:
    """On reference artery (1) / vein (2) pixels: green = right, red = artery called vein, magenta = vein called artery,
    grey = vessel the prediction left unlabelled."""
    p, r = np.asarray(pred_labels), np.asarray(ref_labels)
    out = np.zeros((*r.shape, 3))
    art_p = (p == 1) | (p == 3)
    vein_p = (p == 2) | (p == 3)
    ref_a, ref_v = r == 1, r == 2
    out[ref_a & art_p] = out[ref_v & vein_p] = (0.2, 0.85, 0.3)
    out[ref_a & vein_p & ~art_p] = (1, 0.15, 0.15)
    out[ref_v & art_p & ~vein_p] = (1, 0.2, 1)
    out[(ref_a | ref_v) & ~art_p & ~vein_p] = (0.5, 0.5, 0.5)
    return out


def av_label_rgb(labels: NDArray) -> NDArray[np.float64]:
    """Artery red, vein blue, both (crossing) purple."""
    lab = np.asarray(labels)
    out = np.zeros((*lab.shape, 3))
    out[lab == 1] = (1, 0.15, 0.15)
    out[lab == 2] = (0.2, 0.4, 1)
    out[lab == 3] = (0.75, 0.3, 1)
    return out
