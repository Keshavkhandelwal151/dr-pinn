"""Milestone 12: one evaluation report for every model - accuracy, precision, recall, F1, R2 and heat maps.

Works from files the earlier steps already saved, so it needs NO GPU and NO PyTorch:
  * M5 vessel / artery-vein probabilities  reports/m5_<preset>/probs/<tag>/<id>_vessel.png, <id>_av.png
    scored against DRIVE / RITE expert labels
  * M6 triage predictions                  reports/m6_<preset>/m6_predictions.csv

Output (reports/m12_eval/):
  metrics.json, metrics.md            every number, with a plain-language table
  seg_per_image.csv                   per-photo segmentation and A/V scores
  heatmap_vessel_confusion.png        vessel vs background, counts and shares
  heatmap_av_confusion_<method>.png   artery vs vein, for the original rule and for segment voting
  heatmap_seg_per_image.png           photos x metrics: which photos are weak
  errors_<id>.png                     where on the retina the vessel and A/V calls go wrong
  heatmap_triage_<set>.png            5 x 5 DR-grade confusion per image set

Usage:
  python scripts/m12_evaluate.py --probs-dir reports/m5_full/probs/test --drive-dir data/DRIVE --rite-dir data/RITE \
      --predictions reports/m6_full/m6_predictions.csv
  python scripts/m12_evaluate.py --synthetic /tmp/fake_m12      # software check only, no real data
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drpinn import DISCLAIMER  # noqa: E402
from drpinn.evaluation import (av_error_rgb, av_label_rgb, binary_report, confusion, plot_confusion_heatmap,  # noqa: E402
                               plot_error_maps, plot_metric_heatmap, r2_score, regression_report,
                               report_from_confusion, vessel_error_rgb)
from drpinn.segmentation.av import av_label_map  # noqa: E402
from drpinn.segmentation.data import load_drive_rite  # noqa: E402
from drpinn.segmentation.metrics import av_label_metrics, roc_auc  # noqa: E402
from drpinn.vessel_graph import compare_centrelines, extract_vessel_graph  # noqa: E402

GRADES = ["0 none", "1 mild", "2 moderate", "3 severe", "4 PDR"]
AV_METHODS = ("threshold", "vote")
SEG_HEATMAP_METRICS = ["accuracy", "precision", "recall", "f1", "auc", "av_threshold_accuracy", "av_vote_accuracy",
                       "av_vote_artery_f1", "av_vote_vein_f1"]


# ---------------------------------------------------------------- segmentation

def load_probabilities(probs_dir: Path, image_id: str) -> np.ndarray | None:
    from PIL import Image

    v = probs_dir / f"{image_id}_vessel.png"
    if not v.exists():
        return None
    vessel = np.asarray(Image.open(v), np.float64) / 255.0
    av_path = probs_dir / f"{image_id}_av.png"
    if av_path.exists():
        av = np.asarray(Image.open(av_path).convert("RGB"), np.float64) / 255.0
        return np.stack([vessel, av[..., 0], av[..., 2]])
    return np.stack([vessel, np.zeros_like(vessel), np.zeros_like(vessel)])


def evaluate_segmentation(samples, probs: dict[str, np.ndarray], out_dir: Path, error_maps: int = 3) -> dict:
    rows = []
    pooled_vessel = np.zeros((2, 2), np.int64)
    pooled_av = {m: np.zeros((2, 2), np.int64) for m in AV_METHODS}
    radius_ref, radius_pred = [], []
    for k, s in enumerate(samples):
        prob = probs[s.image_id]
        if prob.shape[1:] != s.fov.shape:
            raise ValueError(f"photo {s.image_id}: probability map {prob.shape[1:]} does not match the label {s.fov.shape}")
        fov = s.fov
        pred = (prob[0] >= 0.5) & fov
        b = binary_report(s.vessel[fov], pred[fov])
        pooled_vessel += np.array([[b["tn"], b["fp"]], [b["fn"], b["tp"]]])
        row = {"image_id": s.image_id, "reference": s.vessel_source,
               **{k2: b[k2] for k2 in ("accuracy", "precision", "recall", "specificity", "f1", "iou",
                                       "balanced_accuracy")},
               "auc": roc_auc(prob[0][fov], s.vessel[fov])}
        try:
            ref_g = extract_vessel_graph(s.vessel, fov=fov)
            pred_g = extract_vessel_graph(pred, fov=fov, probability=prob[0])
            c = compare_centrelines(pred_g.paths_rc, pred_g.radius_px, ref_g.paths_rc, ref_g.radius_px,
                                    mask=pred_g.radius_reliable)
            row.update({"centreline_recall": c.recall, "centreline_precision": c.precision,
                        "radius_r2": r2_score(c.radius_ref, c.radius_test)})
            radius_ref += list(c.radius_ref)
            radius_pred += list(c.radius_test)
        except ValueError as exc:
            row["graph_error"] = str(exc)
        labels = {}
        if s.has_av and prob[1:].any():
            centre = skeletonize(s.vessel)
            ref_a, ref_v = centre & (s.av_labels == 1), centre & (s.av_labels == 2)
            for m in AV_METHODS:
                lab = labels[m] = av_label_map(prob, method=m, vessel=pred)
                row.update(av_label_metrics(lab, s.av_labels, s.vessel, prefix=f"av_{m}"))
                is_a, is_v = (lab == 1) | (lab == 3), (lab == 2) | (lab == 3)
                # rows: true artery / vein; a pixel missing from the right tree counts as the other class
                pooled_av[m] += np.array([[(ref_a & is_a).sum(), (ref_a & ~is_a).sum()],
                                          [(ref_v & ~is_v).sum(), (ref_v & is_v).sum()]])
        rows.append(row)
        if k < error_maps:
            panels = [(vessel_error_rgb(pred, s.vessel, fov), "vessels: white hit · red extra · blue missed")]
            if labels:
                panels += [(av_label_rgb(labels["vote"]), "model A/V (segment vote): artery red · vein blue"),
                           (av_error_rgb(labels["threshold"], s.av_labels), "A/V errors, original rule"),
                           (av_error_rgb(labels["vote"], s.av_labels), "A/V errors, segment vote\n"
                            "green right · red artery→vein · pink vein→artery · grey unlabelled")]
            plot_error_maps(s.rgb, panels, out_dir / f"errors_{s.image_id}.png", f"photo {s.image_id}")

    figures = [plot_confusion_heatmap(pooled_vessel, ["background", "vessel"], out_dir / "heatmap_vessel_confusion.png",
                                      "Vessel segmentation, all test pixels inside the field of view")]
    out = {"photos": len(rows), "vessel_pooled": report_from_confusion(pooled_vessel, ["background", "vessel"]),
           "vessel_mean_over_photos": _mean_rows(rows), "radius": regression_report(radius_ref, radius_pred)}
    if any(m.sum() for m in pooled_av.values()):
        out["artery_vein"] = {}
        for m, cm in pooled_av.items():
            out["artery_vein"][m] = report_from_confusion(cm, ["artery", "vein"])
            figures.append(plot_confusion_heatmap(cm, ["artery", "vein"], out_dir / f"heatmap_av_confusion_{m}.png",
                                                  f"Artery vs vein on the expert centreline - "
                                                  f"{'original rule' if m == 'threshold' else 'segment vote'}"))
    shown = [m for m in SEG_HEATMAP_METRICS if any(np.isfinite(float(r.get(m, np.nan))) for r in rows)]
    figures.append(plot_metric_heatmap(rows, shown, out_dir / "heatmap_seg_per_image.png",
                                       title="Per-photo scores (green good, red poor)"))
    out["figures"] = [str(f) for f in figures]
    _write_csv(rows, out_dir / "seg_per_image.csv")
    return out


def _mean_rows(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))] if rows else []
    out = {}
    for k in keys:
        v = np.array([float(r.get(k, np.nan)) for r in rows])
        v = v[np.isfinite(v)]
        if v.size:
            out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size)}
    return out


# ---------------------------------------------------------------- triage

def evaluate_triage(predictions_csv: Path, out_dir: Path) -> dict:
    with open(predictions_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("true_grade", "") not in ("", None)]
    out = {}
    for name in dict.fromkeys(r["set"] for r in rows):
        rs = [r for r in rows if r["set"] == name]
        y = np.array([int(float(r["true_grade"])) for r in rs])
        pred = np.array([int(float(r["pred_grade"])) for r in rs])
        p = np.array([[float(r[f"p{g}"]) for g in range(5)] for r in rs])
        cm = confusion(y, pred, 5)
        rep = report_from_confusion(cm, GRADES)
        expected = p @ np.arange(5)
        ref_y = y >= 2
        p_ref = np.array([float(r["p_referable"]) for r in rs])
        # M6 marks photos above its validation-fixed referral threshold as "higher priority"
        flagged = np.array([r.get("priority") == "higher priority" for r in rs]) \
            if any(r.get("priority") == "higher priority" for r in rs) else p_ref >= 0.5
        out[name] = {"images": len(rs), "accuracy": rep["accuracy"], "macro_precision": rep["macro_precision"],
                     "macro_recall": rep["macro_recall"], "macro_f1": rep["macro_f1"],
                     "weighted_f1": rep["weighted_f1"], "per_grade": rep["per_class"],
                     "grade_r2": regression_report(y, expected),
                     "referable": {**binary_report(ref_y, p_ref >= 0.5, p_ref), "rule": "p_referable >= 0.5"},
                     "referable_at_operating_point": binary_report(ref_y, flagged),
                     "confusion_matrix": cm.tolist()}
        plot_confusion_heatmap(cm, GRADES, out_dir / f"heatmap_triage_{name}.png",
                               f"DR grade, {name} ({len(rs)} photos)")
    return out


# ---------------------------------------------------------------- report

def _fmt(v) -> str:
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.3f}"


def markdown(res: dict) -> str:
    md = ["# M12 · Model evaluation", "", f"> {DISCLAIMER}", ""]
    if res.get("synthetic"):
        md += ["**SYNTHETIC inputs: software check only, these are not results.**", ""]
    seg = res.get("segmentation")
    if seg:
        v = seg["vessel_pooled"]
        md += ["## Vessel segmentation", f"{seg['photos']} photos, pixels inside the field of view.", "",
               "| accuracy | precision | recall | F1 (Dice) | specificity |", "|---|---|---|---|---|",
               f"| {_fmt(v['accuracy'])} | {_fmt(v['per_class']['vessel']['precision'])} | "
               f"{_fmt(v['per_class']['vessel']['recall'])} | {_fmt(v['per_class']['vessel']['f1'])} | "
               f"{_fmt(v['per_class']['vessel']['specificity'])} |", "",
               "Accuracy is high for any model because most pixels are background; F1 is the number to compare.", "",
               f"Vessel radius, model vs expert (matched segments, n = {seg['radius']['n']}): "
               f"R² = {_fmt(seg['radius']['r2'])}, MAE = {_fmt(seg['radius']['mae'])} px.", ""]
        if "artery_vein" in seg:
            md += ["## Artery vs vein (expert centreline)", "",
                   "| method | accuracy | artery precision | artery recall | artery F1 | vein precision | vein recall | vein F1 |",
                   "|---|---|---|---|---|---|---|---|"]
            for m, r in seg["artery_vein"].items():
                a, w = r["per_class"]["artery"], r["per_class"]["vein"]
                md.append(f"| {'original rule (threshold each map)' if m == 'threshold' else 'segment vote'} | "
                          f"{_fmt(r['accuracy'])} | {_fmt(a['precision'])} | {_fmt(a['recall'])} | {_fmt(a['f1'])} | "
                          f"{_fmt(w['precision'])} | {_fmt(w['recall'])} | {_fmt(w['f1'])} |")
            mv = seg["vessel_mean_over_photos"]
            if "av_threshold_unlabelled" in mv:
                md += ["", f"Share of expert artery/vein centreline left in NEITHER tree: original rule "
                           f"{_fmt(mv['av_threshold_unlabelled']['mean'])}, segment vote "
                           f"{_fmt(mv.get('av_vote_unlabelled', {}).get('mean'))}.", ""]
    tri = res.get("triage")
    if tri:
        md += ["## DR grade triage", "",
               "| set | photos | accuracy | macro precision | macro recall | macro F1 | grade R² | referable AUC | "
               "referable sensitivity | referable specificity |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in tri.items():
            op = r["referable_at_operating_point"]
            md.append(f"| {name} | {r['images']} | {_fmt(r['accuracy'])} | {_fmt(r['macro_precision'])} | "
                      f"{_fmt(r['macro_recall'])} | {_fmt(r['macro_f1'])} | {_fmt(r['grade_r2']['r2'])} | "
                      f"{_fmt(r['referable'].get('auc'))} | {_fmt(op['recall'])} | {_fmt(op['specificity'])} |")
        md += ["", "Grade R²: expected grade (Σ g·p_g) against the true grade. Referable AUC is NaN when a set "
                   "has no non-referable photo (then the set itself needs checking).", ""]
    md += ["## Figures", ""] + [f"- `{Path(f).name}`" for f in res.get("figures", [])]
    return "\n".join(md) + "\n"


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run(out_dir: Path, probs_dir: Path | None = None, drive_dir=None, rite_dir=None,
        predictions: Path | None = None, error_maps: int = 3, synthetic: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"synthetic": synthetic}
    if probs_dir is not None:
        probs_dir = Path(probs_dir)
        samples = load_drive_rite(drive_dir, rite_dir)
        probs = {s.image_id: p for s in samples if (p := load_probabilities(probs_dir, s.image_id)) is not None}
        samples = [s for s in samples if s.image_id in probs]
        if not samples:
            raise FileNotFoundError(f"no <id>_vessel.png probability maps in {probs_dir} that match a labelled photo")
        res["segmentation"] = evaluate_segmentation(samples, probs, out_dir, error_maps)
    if predictions is not None:
        res["triage"] = evaluate_triage(Path(predictions), out_dir)
    res["figures"] = sorted(str(p) for p in out_dir.glob("*.png"))
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    (out_dir / "metrics.md").write_text(markdown(res))
    return res


def make_synthetic(root: Path, seed: int = 0) -> tuple[Path, Path, Path, Path]:
    """FAKE labelled photos, imperfect 'model' outputs and triage predictions (software check only)."""
    from PIL import Image
    from scipy import ndimage

    from drpinn.datasets.synthetic import make_fake_drive_rite

    make_fake_drive_rite(root, n_train=2, n_test=4, size=256, seed=seed)
    drive, rite = root / "DRIVE", root / "RITE"
    probs_dir = root / "probs" / "test"
    probs_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for s in load_drive_rite(drive, rite):
        if s.split != "test":
            continue
        shape = s.fov.shape

        def smooth(sd, shape=shape):
            return ndimage.gaussian_filter(rng.normal(0, 1, shape), sd)
        vessel = np.clip(ndimage.gaussian_filter(s.vessel.astype(float), 1.0) + 0.6 * smooth(2), 0, 1) * s.fov
        lab = s.av_labels
        sig = np.where(lab == 1, 0.6, np.where(lab == 2, -0.6, 0.0))
        d = 3 * (sig + 3.0 * smooth(2))
        pa, pv = 0.9 / (1 + np.exp(-d)), 0.9 / (1 + np.exp(d))
        Image.fromarray((vessel * 255).astype(np.uint8)).save(probs_dir / f"{s.image_id}_vessel.png")
        av = np.zeros((*s.fov.shape, 3), np.uint8)
        av[..., 0], av[..., 2] = (pa * 255).astype(np.uint8), (pv * 255).astype(np.uint8)
        Image.fromarray(av).save(probs_dir / f"{s.image_id}_av.png")
    pred_csv = root / "m6_predictions.csv"
    rows = []
    for name, n in (("test_internal", 300), ("external_idrid_test", 100)):
        y = rng.choice(5, n, p=[0.49, 0.10, 0.27, 0.05, 0.09])
        logits = np.eye(5)[y] * rng.uniform(1.0, 3.5, (n, 1)) + rng.normal(0, 1.0, (n, 5))
        p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        for i in range(n):
            pr = p[i, 2:].sum()
            rows.append({"set": name, "dataset": "fake", "image_id": f"{name}_{i}", "true_grade": int(y[i]),
                         "pred_grade": int(p[i].argmax()), **{f"p{g}": round(float(p[i, g]), 4) for g in range(5)},
                         "p_referable": round(float(pr), 4), "priority": "higher priority" if pr >= 0.4 else "lower priority"})
    _write_csv(rows, pred_csv)
    return probs_dir, drive, rite, pred_csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probs-dir", default=None, help="saved M5 probabilities, e.g. reports/m5_full/probs/test")
    ap.add_argument("--drive-dir", default=str(ROOT / "data" / "DRIVE"))
    ap.add_argument("--rite-dir", default=str(ROOT / "data" / "RITE"))
    ap.add_argument("--predictions", default=None, help="M6 predictions, e.g. reports/m6_full/m6_predictions.csv")
    ap.add_argument("--out", default=str(ROOT / "reports" / "m12_eval"))
    ap.add_argument("--error-maps", type=int, default=3)
    ap.add_argument("--synthetic", default=None, help="write FAKE inputs here and evaluate them (software check)")
    a = ap.parse_args()
    if a.synthetic:
        probs, drive, rite, pred = make_synthetic(Path(a.synthetic))
        print("SYNTHETIC inputs - software check only, not results", flush=True)
        run(Path(a.out), probs, drive, rite, pred, a.error_maps, synthetic=True)
    else:
        if a.probs_dir is None and a.predictions is None:
            ap.error("give --probs-dir and/or --predictions (or --synthetic)")
        run(Path(a.out), a.probs_dir, a.drive_dir, a.rite_dir, a.predictions, a.error_maps)
    print((Path(a.out) / "metrics.md").read_text())


if __name__ == "__main__":
    main()
