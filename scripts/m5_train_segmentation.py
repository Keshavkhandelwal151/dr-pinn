"""M5: train and evaluate the vessel (+ artery/vein) U-Net.

  cv     5-fold cross-validation on the 20 annotated DRIVE training photos
         (folds fixed per photo: drpinn.datasets.splits.drive_fold). Reports
         Dice/IoU/Se/Sp/accuracy/AUC, centreline agreement with the graph of
         the expert mask, and - with RITE - artery/vein accuracy and the
         artery/vein graph separation.
  final  one model on all 20 training photos (the model later milestones use);
         with RITE, tested on the 20 DRIVE test photos against RITE's vessel
         reference (DRIVE's second observer) and A/V labels.

Checkpoints are saved during training and resumed automatically.

    python scripts/m5_train_segmentation.py --drive-dir data/DRIVE --rite-dir data/RITE --mode both
    python scripts/m5_train_segmentation.py --synthetic /tmp/fake_m5 --preset tiny   # software test only
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from drpinn.datasets.splits import drive_fold  # noqa: E402
from drpinn.evaluation import r2_score  # noqa: E402
from drpinn.segmentation.av import av_graphs, av_label_map, masks_from_probabilities  # noqa: E402
from drpinn.segmentation.data import load_drive_rite  # noqa: E402
from drpinn.segmentation.infer import predict_image  # noqa: E402
from drpinn.segmentation.metrics import av_label_metrics, av_metrics, segmentation_metrics  # noqa: E402
from drpinn.segmentation.train import TrainConfig, train, write_json  # noqa: E402
from drpinn.segmentation.unet import UNetConfig, count_parameters  # noqa: E402
from drpinn.vessel_graph import compare_centrelines, extract_vessel_graph  # noqa: E402

PRESETS = {
    "full": TrainConfig(iterations=4000, batch=16, patch=96, save_every=1000, unet=UNetConfig(base=32, depth=4)),
    # Artery/vein needs wider context than vessel detection (a vessel is told apart by following it,
    # by its width and colour relative to its neighbours): bigger patches, longer training, balanced A/V loss.
    "av": TrainConfig(iterations=10000, batch=12, patch=192, save_every=1000, vessel_fraction=0.7,
                      av_loss_weight=2.0, av_balance=True, unet=UNetConfig(base=32, depth=4)),
    "quick": TrainConfig(iterations=400, batch=16, patch=96, unet=UNetConfig(base=16, depth=4)),
    "tiny": TrainConfig(iterations=300, batch=8, patch=64, save_every=100,
                        unet=UNetConfig(base=8, depth=3, dropout=0.0)),
}
TILE = {"av": 384, "full": 256, "quick": 256, "tiny": 128}


def evaluate(model, sample, device, tile, use_av, out_dir: Path, tag: str, overlay: bool) -> dict:
    prob = predict_image(model, sample.image, sample.fov, tile=tile, device=device)
    row = {"image_id": sample.image_id, "reference": sample.vessel_source, **segmentation_metrics(prob[0], sample.vessel, sample.fov)}
    row["f1"] = row["dice"]
    pred = (prob[0] >= 0.5) & sample.fov
    try:
        ref_g = extract_vessel_graph(sample.vessel, fov=sample.fov)
        pred_g = extract_vessel_graph(pred, fov=sample.fov, probability=prob[0])
        c = compare_centrelines(pred_g.paths_rc, pred_g.radius_px, ref_g.paths_rc, ref_g.radius_px,
                                mask=pred_g.radius_reliable)
        row.update({"centreline_recall": c.recall, "centreline_precision": c.precision,
                    "radius_median_abs_error": c.radius_median_abs_error, "radius_r2": r2_score(c.radius_ref, c.radius_test),
                    "segments_pred": pred_g.n_edges, "segments_ref": ref_g.n_edges})
    except ValueError as exc:  # e.g. an empty prediction
        row["graph_error"] = str(exc)
    if use_av and sample.has_av:
        row.update(av_metrics(prob[1], prob[2], sample.av_labels, sample.vessel))
        for method in ("threshold", "vote"):  # how the A/V maps are turned into the two trees
            lab = av_label_map(prob, method=method, vessel=(prob[0] >= 0.5) & sample.fov)
            row.update(av_label_metrics(lab, sample.av_labels, sample.vessel, prefix=f"av_{method}"))
    if use_av:
        v, a, w = masks_from_probabilities(prob)
        try:
            row.update(av_graphs(v & sample.fov, a, w, fov=sample.fov, vessel_prob=prob[0]).summary())
        except ValueError as exc:
            row["av_graph_error"] = str(exc)
    save_probabilities(prob, out_dir / "probs" / tag / sample.image_id, use_av)
    if overlay:
        save_overlay(sample, prob, use_av, out_dir / "overlays" / f"{tag}_{sample.image_id}.png")
    return row


def save_probabilities(prob: np.ndarray, stem: Path, use_av: bool) -> None:
    from PIL import Image

    stem.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(prob[0], 0, 1) * 255).round().astype(np.uint8)).save(f"{stem}_vessel.png")
    if use_av:
        av = np.zeros((*prob.shape[1:], 3), np.uint8)
        av[..., 0] = (np.clip(prob[1], 0, 1) * 255).round()
        av[..., 2] = (np.clip(prob[2], 0, 1) * 255).round()
        Image.fromarray(av).save(f"{stem}_av.png")


def save_overlay(sample, prob, use_av, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = (prob[0] >= 0.5) & sample.fov
    err = np.zeros((*pred.shape, 3))
    err[pred & sample.vessel] = (1, 1, 1)
    err[pred & ~sample.vessel] = (1, 0.2, 0.2)
    err[~pred & sample.vessel] = (0.2, 0.5, 1)
    panels = [(sample.rgb, f"photo {sample.image_id}"), (sample.vessel, "expert"),
              (prob[0], "model probability"), (err, "white hit · red extra · blue missed")]
    if use_av:
        av = np.zeros((*pred.shape, 3))
        av[..., 0] = pred * (prob[1] >= 0.5)
        av[..., 2] = pred * (prob[2] >= 0.5)
        panels.append((av, "model: artery red · vein blue"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.6))
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im, cmap="gray" if np.ndim(im) == 2 else None)
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def summarise(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and k != "image_id"] if rows else []
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if isinstance(r.get(k), (int, float))], float)
        v = v[np.isfinite(v)]
        if v.size:
            out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size)}
    return out


def write_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run(drive_dir, rite_dir, out_dir, ckpt_dir, mode="both", folds=(0, 1, 2, 3, 4), preset="full",
        device=None, overlays=2, iterations=None, seed=0) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir, ckpt_dir = Path(out_dir), Path(ckpt_dir)
    cfg = replace(PRESETS[preset], seed=seed)
    if iterations:
        cfg = replace(cfg, iterations=int(iterations))
    t_load = time.perf_counter()
    samples = load_drive_rite(drive_dir, rite_dir)
    train_pool = [s for s in samples if s.split == "train"]
    test_pool = [s for s in samples if s.split == "test"]
    if not train_pool:
        raise FileNotFoundError(f"no annotated DRIVE training photos found under {drive_dir}")
    use_av = any(s.has_av for s in train_pool)
    env = {"device": device, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
           "torch": torch.__version__, "python": platform.python_version()}
    print(f"{len(train_pool)} annotated training photos, {len(test_pool)} test photos; "
          f"artery/vein labels: {'yes (RITE)' if use_av else 'no (RITE not found) - vessel model only'}; "
          f"device {device}{' (' + env['gpu'] + ')' if env['gpu'] else ''}; loaded in {time.perf_counter() - t_load:.0f} s", flush=True)
    results = {"preset": preset, "config": cfg.to_dict(), "environment": env, "use_av": use_av,
               "train_photos": [s.image_id for s in train_pool], "test_photos": [s.image_id for s in test_pool]}

    if mode in ("cv", "both"):
        rows = []
        for k in folds:
            val = [s for s in train_pool if drive_fold(s.image_id) == k]
            trn = [s for s in train_pool if drive_fold(s.image_id) != k]
            if not val or not trn:
                print(f"fold {k}: skipped (no {'validation' if not val else 'training'} photos)")
                continue
            print(f"fold {k}: training on {len(trn)}, validating on {[s.image_id for s in val]}", flush=True)
            t0 = time.perf_counter()
            model, ck = train(trn, replace(cfg, seed=seed * 10 + k), ckpt_dir / f"fold{k}.pt", device,
                              meta={"fold": k, "val_ids": [s.image_id for s in val]}, use_av=use_av)
            for j, s in enumerate(val):
                rows.append({"fold": k, **evaluate(model, s, device, TILE[preset], use_av, out_dir, f"cv_fold{k}",
                                                   overlay=j < overlays)})
            print(f"fold {k}: done in {time.perf_counter() - t0:.0f} s; "
                  f"Dice {np.mean([r['dice'] for r in rows if r['fold'] == k]):.3f}", flush=True)
        results["cv"] = {"per_image": rows, "summary": summarise(rows)}
        write_rows(rows, out_dir / "cv_per_image.csv")

    if mode in ("final", "both"):
        print(f"final model: training on all {len(train_pool)} annotated photos", flush=True)
        model, ck = train(train_pool, cfg, ckpt_dir / "final.pt", device, meta={"fold": "all"}, use_av=use_av)
        results["final_model"] = {"checkpoint": str(ckpt_dir / "final.pt"), "parameters": count_parameters(model)}
        if test_pool:
            rows = [evaluate(model, s, device, TILE[preset], use_av, out_dir, "test", overlay=j < overlays)
                    for j, s in enumerate(test_pool)]
            results["test"] = {"per_image": rows, "summary": summarise(rows),
                               "note": "held-out DRIVE test photos; vessel reference = DRIVE 2nd observer (via RITE)"}
            write_rows(rows, out_dir / "test_per_image.csv")
        else:
            results["test"] = {"note": "no test photos (needs RITE): only cross-validation results are available"}
    write_json(results, out_dir / "m5_results.json")
    return results


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-dir", default=str(root / "data" / "DRIVE"))
    ap.add_argument("--rite-dir", default=str(root / "data" / "RITE"))
    ap.add_argument("--out", default=str(root / "reports" / "m5"))
    ap.add_argument("--ckpt-dir", default=str(root / "checkpoints" / "m5"))
    ap.add_argument("--mode", choices=["cv", "final", "both"], default="both")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--preset", choices=list(PRESETS), default="full")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--overlays", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", default=None, help="write FAKE DRIVE+RITE here first (software test only)")
    a = ap.parse_args()
    drive, rite = a.drive_dir, a.rite_dir
    if a.synthetic:
        from drpinn.datasets.synthetic import make_fake_drive_rite
        make_fake_drive_rite(a.synthetic)
        drive, rite = str(Path(a.synthetic) / "DRIVE"), str(Path(a.synthetic) / "RITE")
        print("SYNTHETIC data — software test only, not results", flush=True)
    r = run(drive, rite, a.out, a.ckpt_dir, a.mode, tuple(int(x) for x in a.folds.split(",")), a.preset,
            iterations=a.iterations, overlays=a.overlays, seed=a.seed)
    for part in ("cv", "test"):
        s = r.get(part, {}).get("summary")
        if s:
            line = "  ".join(f"{k} {s[k]['mean']:.3f}±{s[k]['sd']:.3f}" for k in
                             ("dice", "auc", "sensitivity", "specificity", "centreline_recall",
                              "radius_median_abs_error", "av_accuracy", "av_vote_accuracy") if k in s)
            print(f"{part.upper():5s} (n={s['dice']['n']}): {line}")


if __name__ == "__main__":
    main()
