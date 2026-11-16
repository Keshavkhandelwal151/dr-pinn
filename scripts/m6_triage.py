"""M6: DR grade triage model: train (resumable), calibrate on validation, test internally and externally.

Data (rows of data/splits/manifest.csv, written by M4):
  train / val / internal test   APTOS 2019, manifest split train / val / test (70 / 15 / 15)
  external test (PRIMARY)       IDRiD grading, official TEST images (different camera and clinic)
  external (secondary)          IDRiD grading TRAIN images (never used for training here)
IDRiD is never trained on, so both IDRiD sets are external. Any test image that
shares a duplicate group with a training or validation image is dropped (the
manifest already prevents this; this is a second check).

Pre-registered (docs/m6_analysis_plan.md), fixed before any real-data run:
  model selection    epoch with the highest validation QWK
  calibration        one temperature, fitted on validation
  referral threshold calibrated P(grade >= 2) giving >= 90 % sensitivity on validation
  uncertainty limit  the 10 % most uncertain validation photos (normalised entropy)
  PRIMARY            referable AUC on the external IDRiD test; QWK on the internal APTOS test
  SECONDARY          sensitivity / specificity at the fixed threshold, macro-F1, per-grade
                     results, calibration error, selective prediction, quality-flag strata

    python scripts/m6_triage.py --data-dir data --preset full
    python scripts/m6_triage.py --synthetic --preset tiny          # software check, no data needed
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from drpinn import DISCLAIMER, __version__  # noqa: E402
from drpinn.datasets.records import REFERABLE_MIN_GRADE  # noqa: E402
from drpinn.segmentation.metrics import roc_auc  # noqa: E402
from drpinn.triage.data import build_cache, load_cached, synthetic_graded_eye, synthetic_grades, to_input  # noqa: E402
from drpinn.triage.metrics import (  # noqa: E402
    expected_calibration_error, fit_temperature, nll, normalised_entropy, quadratic_weighted_kappa,
    referable_probability, reliability_bins, roc_curve, softmax, summarise, top_label_ece,
)
from drpinn.triage.model import grad_cam  # noqa: E402
from drpinn.triage.quality import QualityReference  # noqa: E402
from drpinn.triage.report import choose_operating_point, triage_card  # noqa: E402
from drpinn.triage.train import TriageConfig, load_triage, predict_logits, train_triage  # noqa: E402

PRESETS = {
    "full": TriageConfig(arch="mobilenet_v3_large", pretrained=True, image_size=512, epochs=20, batch=32, lr=5e-4),
    "quick": TriageConfig(arch="mobilenet_v3_large", pretrained=True, image_size=384, epochs=6, batch=32, lr=5e-4),
    "tiny": TriageConfig(arch="tiny", pretrained=False, image_size=128, epochs=20, batch=16, lr=3e-3),
}
TARGET_SENSITIVITY = 0.90
UNCERTAIN_SHARE = 0.10
SETS = {"val": "APTOS validation (used to calibrate)", "test_internal": "APTOS internal test",
        "external_idrid_test": "IDRiD official test (external, PRIMARY)",
        "external_idrid_train": "IDRiD train images (external, secondary)"}


# --- data -----------------------------------------------------------------------------

def manifest_sha(data_dir: Path) -> str:
    """Short hash of the manifest file, so results can be matched to the exact split they used."""
    import hashlib

    return hashlib.sha256((Path(data_dir) / "splits" / "manifest.csv").read_bytes()).hexdigest()[:16]


def read_manifest(data_dir: Path) -> list[dict]:
    path = data_dir / "splits" / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run M4 (notebook step 11d) first")
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["part"] != "grading" or r["dataset"] not in ("APTOS2019", "IDRiD") or r["dr_grade"] == "":
                continue
            if r.get("label_conflict", "0") == "1":
                continue
            p = Path(r["rel_path"])
            rows.append({**r, "path": p if p.is_absolute() else data_dir / p, "dr_grade": int(r["dr_grade"])})
    return rows


def make_sets(rows: list[dict]) -> tuple[dict[str, list[dict]], dict]:
    sets = {"train": [], "val": [], "test_internal": [], "external_idrid_test": [], "external_idrid_train": []}
    for r in rows:
        if r["dataset"] == "APTOS2019":
            name = {"train": "train", "val": "val", "test": "test_internal"}.get(r["split"])
        else:
            # the official IDRiD test set (pre-registered), not photos the manifest moved to "test" as duplicates
            official = r.get("official_split") or r["split"]
            name = "external_idrid_test" if official == "test" else "external_idrid_train"
        if name:
            sets[name].append(r)
    used = {r["dup_group"] for r in sets["train"] + sets["val"]}
    dropped = {}
    for name in ("test_internal", "external_idrid_test", "external_idrid_train"):
        keep = [r for r in sets[name] if r["dup_group"] not in used]
        dropped[name] = len(sets[name]) - len(keep)
        sets[name] = keep
    return sets, {"dropped_as_duplicates_of_training": dropped}


def synthetic_manifest(root: Path, n: int, scenario: str, seed: int = 0, size: int = 192) -> list[dict]:
    """Fake APTOS-like and IDRiD-like sets (IDRiD-like: other tint and resolution). SOFTWARE CHECK ONLY."""
    from PIL import Image

    if scenario not in ("planted", "null"):
        raise ValueError("scenario must be 'planted' or 'null'")
    rng = np.random.default_rng(seed + 7)
    n_ext = max(40, n // 5)
    plan = ([("APTOS2019", s) for s in rng.choice(["train", "val", "test"], n, p=[0.7, 0.15, 0.15])]
            + [("IDRiD", "test")] * n_ext + [("IDRiD", "train")] * (n_ext // 2))
    grades = synthetic_grades(len(plan), seed)
    rows = []
    for k, ((ds, split), g) in enumerate(zip(plan, grades)):
        lesion_grade = int(rng.choice(5, p=[0.49, 0.10, 0.27, 0.05, 0.09])) if scenario == "null" else None
        tint, s = ((1.0, 1.0, 1.0), size) if ds == "APTOS2019" else ((1.08, 0.9, 0.8), int(size * 1.5))
        path = root / ds / f"{ds}_{k:05d}.png"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(synthetic_graded_eye(int(g), s, seed * 100003 + k, lesion_grade, tint)).save(path)
        rows.append({"dataset": ds, "part": "grading", "image_id": f"{k:05d}", "split": split, "official_split": split,
                     "dr_grade": int(g), "dup_group": str(k), "path": path})
    return rows


def _key(r: dict) -> str:
    return f"{r['dataset']}_{r['image_id']}"


def _finished_training(ckpt_dir: Path, cfg: TriageConfig) -> tuple[dict, dict] | None:
    """(meta, state) of a COMPLETE training run with these settings, else None."""
    last = Path(ckpt_dir) / "last.pt"
    if not last.exists():
        return None
    ck = torch.load(last, map_location="cpu", weights_only=False)
    if ck.get("config") != cfg.to_dict() or not ck["state"].get("complete"):
        return None
    return ck.get("meta", {}), ck["state"]


def _check_no_leak(meta: dict, sets: dict[str, list[dict]], ckpt_dir: Path) -> None:
    """Refuse to evaluate if any photo the model was trained on is now a validation or test photo."""
    if "train_keys" in meta:
        evaluated = {_key(r) for k, rs in sets.items() if k != "train" for r in rs}
        leaked = set(meta["train_keys"]) & evaluated
        if leaked:
            raise RuntimeError(f"{len(leaked)} photos this model was trained on are now validation/test photos "
                               f"(the splits changed, e.g. {sorted(leaked)[0]}). Delete {ckpt_dir} and retrain.")
        return
    # checkpoints from 0.7.0-0.7.1 store only set sizes
    trained, now = meta.get("sets", {}), {k: len(v) for k, v in sets.items()}
    if trained.get("train") != now["train"] or trained.get("val") != now["val"]:
        raise RuntimeError(f"the APTOS split changed since this model was trained (train/val {trained.get('train')}/"
                           f"{trained.get('val')} then, {now['train']}/{now['val']} now). Delete {ckpt_dir} and retrain.")
    print("  note: this checkpoint stores only set sizes (they match). The photo-level check is the 'APTOS vs "
          "previous manifest' line of step 11d.", flush=True)


# --- analysis -------------------------------------------------------------------------

def evaluate_set(logits, y, temperature, op, flags, n_boot, seed) -> dict:
    raw, cal = softmax(logits, 1.0), softmax(logits, temperature)
    s = summarise(cal, y, op.referral_threshold, n_boot, seed)
    ref_y = y >= REFERABLE_MIN_GRADE
    s["calibration"] = {
        "nll_before": nll(logits, y, 1.0), "nll_after": nll(logits, y, temperature),
        "top_label_ece_before": top_label_ece(raw, y), "top_label_ece_after": top_label_ece(cal, y),
        "referable_ece_before": expected_calibration_error(referable_probability(raw), ref_y, 10),
        "referable_ece_after": expected_calibration_error(referable_probability(cal), ref_y, 10)}
    unc = normalised_entropy(cal)
    s["uncertain_share"] = float((unc > op.uncertainty_limit).mean())
    flagged = np.array([bool(f) for f in flags])
    strata = {}
    for name, m in (("quality_ok", ~flagged), ("quality_flagged", flagged)):
        strata[name] = {"images": int(m.sum()),
                        "qwk": quadratic_weighted_kappa(y[m], cal[m].argmax(1)) if m.any() else float("nan"),
                        "referable_auc": roc_auc(referable_probability(cal[m]), ref_y[m]) if m.any() else float("nan")}
    s["quality"] = {"flag_counts": dict(Counter(f for fl in flags for f in fl)), **strata}
    return s


def headline(results: dict) -> list[str]:
    def fmt(m):
        return f"{m['value']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]"

    sets, out = results["sets"], []
    if "external_idrid_test" in sets:
        e = sets["external_idrid_test"]
        out.append(f"PRIMARY  referable AUC, IDRiD external test (n={e['images']}): {fmt(e['referable']['auc'])}")
        out.append(f"         at the validation threshold: sensitivity {fmt(e['referable']['sensitivity'])}, "
                   f"specificity {fmt(e['referable']['specificity'])}")
    else:
        out.append("PRIMARY  IDRiD external test: not available (no IDRiD grading rows in the manifest)")
    if "test_internal" in sets:
        t = sets["test_internal"]
        out.append(f"PRIMARY  QWK, APTOS internal test (n={t['images']}): {fmt(t['qwk'])}; "
                   f"referable AUC {fmt(t['referable']['auc'])}")
    return out


# --- figures --------------------------------------------------------------------------

def _figures(out_dir: Path, results: dict, preds: dict, images: dict, model, device, op) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    h = results["training"]["history"]
    if h:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        ep = [r["epoch"] for r in h]
        ax[0].plot(ep, [r["train_loss"] for r in h], label="train (augmented)")
        ax[0].plot(ep, [r["val_loss"] for r in h], label="validation")
        ax[0].set(xlabel="epoch", ylabel="loss")
        ax[0].legend()
        ax[1].plot(ep, [r["val_qwk"] for r in h], color="C2")
        ax[1].axvline(results["training"]["best_epoch"], color="0.5", ls=":", label="selected epoch")
        ax[1].set(xlabel="epoch", ylabel="validation QWK")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "m6_training.png", dpi=120)
        plt.close(fig)
        made.append("m6_training.png")

    names = [n for n in ("test_internal", "external_idrid_test", "external_idrid_train") if n in results["sets"]]
    if names:
        fig, axs = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.4), squeeze=False)
        for ax, n in zip(axs[0], names):
            cm = np.array(results["sets"][n]["confusion_matrix"], float)
            norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if norm[i, j] > 0.6 else "black")
            ax.set(xticks=range(5), yticks=range(5), xlabel="predicted grade", ylabel="true grade",
                   title=SETS[n].split(" (")[0])
        fig.tight_layout()
        fig.savefig(out_dir / "m6_confusion.png", dpi=120)
        plt.close(fig)
        made.append("m6_confusion.png")

        fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
        for k, n in enumerate(names):
            p, y = preds[n]
            fpr, tpr = roc_curve(referable_probability(p), y >= REFERABLE_MIN_GRADE)
            r = results["sets"][n]["referable"]
            ax[0].plot(fpr, tpr, color=f"C{k}", label=f"{SETS[n].split(' (')[0]}  AUC {r['auc']['value']:.3f}")
            ax[0].plot(1 - r["specificity"]["value"], r["sensitivity"]["value"], "o", color=f"C{k}")
            b = reliability_bins(referable_probability(p), y >= REFERABLE_MIN_GRADE, 10)
            ax[1].plot([x["mean_confidence"] for x in b], [x["observed"] for x in b], "o-", color=f"C{k}",
                       label=SETS[n].split(" (")[0])
        ax[0].plot([0, 1], [0, 1], color="0.7", ls=":")
        ax[0].set(xlabel="1 - specificity", ylabel="sensitivity", title="Referable DR (dot = validation threshold)")
        ax[0].legend(fontsize=7, loc="lower right")
        ax[1].plot([0, 1], [0, 1], color="0.7", ls=":")
        ax[1].set(xlabel="calibrated P(referable)", ylabel="observed share referable", title="Reliability")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "m6_roc_reliability.png", dpi=120)
        plt.close(fig)
        made.append("m6_roc_reliability.png")

    # Grad-CAM: most confident correct referable / non-referable, and the most confident miss and false alarm
    n = next((s for s in ("external_idrid_test", "test_internal") if s in preds), None)
    if n is not None and model is not None:
        p, y = preds[n]
        pr, ry = referable_probability(p), y >= REFERABLE_MIN_GRADE
        picks = []
        for label, m, fn in (("correct referable", ry & (pr >= op.referral_threshold), np.argmax),
                             ("correct non-referable", ~ry & (pr < op.referral_threshold), np.argmin),
                             ("missed referable", ry & (pr < op.referral_threshold), np.argmin),
                             ("false alarm", ~ry & (pr >= op.referral_threshold), np.argmax)):
            idx = np.flatnonzero(m)
            if idx.size:
                picks.append((label, int(idx[fn(pr[idx])])))
        if picks:
            x = to_input(images[n][[i for _, i in picks]], device)
            cams = grad_cam(model, x, "referable")
            fig, axs = plt.subplots(2, len(picks), figsize=(2.8 * len(picks), 5.8), squeeze=False)
            for k, (label, i) in enumerate(picks):
                axs[0, k].imshow(images[n][i])
                axs[0, k].set_title(f"{label}\ntrue {y[i]}, pred {p[i].argmax()}, P(ref) {pr[i]:.2f}", fontsize=8)
                axs[1, k].imshow(images[n][i])
                axs[1, k].imshow(cams[k], cmap="jet", alpha=0.4)
                axs[1, k].set_title("Grad-CAM (referable)", fontsize=8)
                for a in axs[:, k]:
                    a.axis("off")
            fig.suptitle(f"{SETS[n]} - research prototype, not for clinical use", fontsize=9)
            fig.tight_layout()
            fig.savefig(out_dir / "m6_gradcam.png", dpi=110)
            plt.close(fig)
            made.append("m6_gradcam.png")
    return made


# --- main pipeline --------------------------------------------------------------------

def run(out_dir: Path, data_dir: Path | None = None, ckpt_dir: Path | None = None, cache_dir: Path | None = None,
        cfg: TriageConfig = PRESETS["full"], preset: str = "full", device: str | None = None,
        synthetic: bool = False, scenario: str = "planted", n_synthetic: int = 400,
        stop_after_epochs: int | None = None, n_boot: int = 2000, seed: int = 0, workers: int = 1,
        figures: bool = True) -> dict:
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(ckpt_dir or out_dir / "checkpoint")
    cache_dir = Path(cache_dir or out_dir / "cache") / f"{cfg.image_size}px"
    if synthetic:
        rows = synthetic_manifest(out_dir / "synthetic_data", n_synthetic, scenario, seed)
        source = f"SYNTHETIC ({scenario}) - software check only, not a result"
    else:
        rows = read_manifest(Path(data_dir))
        source = "APTOS 2019 + IDRiD grading (manifest)"
    sets, set_info = make_sets(rows)
    if not sets["train"] or not sets["val"]:
        raise FileNotFoundError("no APTOS train/val rows in the manifest: make the APTOS working copy (step 11b) "
                                "and rebuild the splits (step 11d)")
    done = _finished_training(ckpt_dir, cfg)
    qfile = out_dir / "m6_quality_metrics.json"  # kept in Drive: a new session need not re-read training photos
    saved_q = json.loads(qfile.read_text()) if qfile.exists() else {}
    skip_train = done is not None and all(_key(r) in saved_q for r in sets["train"])
    todo = [r for k, rs in sets.items() if not (skip_train and k == "train") for r in rs]
    print(f"Caching {len(todo)} photos at {cfg.image_size} px in {cache_dir} …"
          + (" (model already trained: training photos not needed)" if skip_train else ""), flush=True)
    new_q, errors = build_cache([(_key(r), r["path"]) for r in todo], cfg.image_size, cache_dir,
                                workers=workers, progress_every=500)
    quality = {**saved_q, **new_q}
    qfile.write_text(json.dumps(quality))
    if errors:
        print(f"  {len(errors)} photos could not be read and are left out, e.g. {next(iter(errors.items()))}")
    sets = {k: [r for r in v if _key(r) not in errors] for k, v in sets.items()}
    if done is not None:
        _check_no_leak(done[0], sets, ckpt_dir)
    images = {k: load_cached([_key(r) for r in v], cache_dir) for k, v in sets.items()
              if v and not (skip_train and k == "train")}
    labels = {k: np.array([r["dr_grade"] for r in v], int) for k, v in sets.items() if v}
    print("  " + ", ".join(f"{k}: {len(v)}" for k, v in sets.items()), flush=True)

    meta = {"source": source, "preset": preset, "sets": {k: len(v) for k, v in sets.items()},
            "train_keys": [_key(r) for r in sets["train"]], "val_keys": [_key(r) for r in sets["val"]]}
    if done is not None:
        state = done[1]
        print(f"  model already trained ({state['epoch']} epochs, selected epoch {state['best_epoch']}): evaluating only")
    else:
        state = train_triage(images["train"], labels["train"], images["val"], labels["val"], cfg, ckpt_dir, device,
                             stop_after_epochs, meta)
    base = {"source": source, "drpinn_version": __version__, "preset": preset, "config": cfg.to_dict(),
            "manifest_sha": None if synthetic else manifest_sha(Path(data_dir)),
            "device": device, "disclaimer": DISCLAIMER}
    if not state["complete"]:
        status = {**base, "status": "incomplete", "epochs_done": state["epoch"], "epochs": cfg.epochs}
        (out_dir / "m6_status.json").write_text(json.dumps(status, indent=2))
        print(f"Stopped after epoch {state['epoch']}/{cfg.epochs}; run again to resume.")
        return status

    model, ck = load_triage(ckpt_dir / "best.pt", device)
    logits = {k: predict_logits(model, x, device) for k, x in images.items() if k != "train"}
    temperature = fit_temperature(logits["val"], labels["val"])
    op = choose_operating_point(logits["val"], labels["val"], temperature, TARGET_SENSITIVITY, UNCERTAIN_SHARE)
    qref = QualityReference.fit([quality[_key(r)] for r in sets["train"]])
    flags = {k: [qref.flags(quality[_key(r)]) for r in v] for k, v in sets.items() if v}

    results = {**base, "status": "complete", "split_info": set_info, "set_sizes": {k: len(v) for k, v in sets.items()},
               "training": {"best_epoch": state["best_epoch"], "best_val_qwk": state["best_val_qwk"],
                            "history": state["history"]},
               "temperature": temperature, "operating_point": op.to_dict(), "quality_reference": qref.to_dict(),
               "sets": {}}
    preds, rows_out = {}, []
    for name in SETS:
        if name not in logits:
            continue
        y = labels[name]
        results["sets"][name] = {"description": SETS[name],
                                 **evaluate_set(logits[name], y, temperature, op, flags[name], n_boot, seed)}
        cal = softmax(logits[name], temperature)
        preds[name] = (cal, y)
        for r, lg, fl in zip(sets[name], logits[name], flags[name]):
            c = triage_card(lg, op, fl)
            rows_out.append({"set": name, "dataset": r["dataset"], "image_id": r["image_id"], "true_grade": r["dr_grade"],
                             "pred_grade": c["grade"], **{f"p{g}": round(v, 4) for g, v in enumerate(c["grade_probabilities"])},
                             "p_referable": round(c["p_referable"], 4), "uncertainty": round(c["uncertainty"], 4),
                             "priority": c["priority"], "notes": "; ".join(c["notes"]), "quality_flags": "; ".join(fl)})
    ex_set = next((s for s in ("external_idrid_test", "test_internal") if s in logits), "val")
    results["example_cards"] = [{"image_id": sets[ex_set][i]["image_id"], "true_grade": int(labels[ex_set][i]),
                                 **triage_card(logits[ex_set][i], op, flags[ex_set][i])} for i in range(min(3, len(labels[ex_set])))]
    results["headline"] = headline(results)
    results["figures"] = _figures(out_dir, results, preds, images, model, device, op) if figures else []
    results["seconds"] = round(time.perf_counter() - t0, 1)
    with open(out_dir / "m6_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    (out_dir / "m6_results.json").write_text(json.dumps(results, indent=2, default=float))
    for line in results["headline"]:
        print(line)
    return results


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(root / "data"))
    ap.add_argument("--out", default=None, help="results folder (default reports/m6_<preset>)")
    ap.add_argument("--ckpt-dir", default=None, help="checkpoint folder (default checkpoints/m6_<preset>)")
    ap.add_argument("--cache-dir", default=None, help="resized-image cache (default /content/m6_cache on Colab)")
    ap.add_argument("--preset", choices=list(PRESETS), default="full")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--stop-after-epochs", type=int, default=None, help="end this run early; run again to resume")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--synthetic", action="store_true", help="fake graded eyes: software check only")
    ap.add_argument("--scenario", choices=["planted", "null"], default="planted")
    ap.add_argument("--n-synthetic", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = PRESETS[a.preset] if a.epochs is None else replace(PRESETS[a.preset], epochs=a.epochs)
    tag = f"{a.preset}{'_synthetic_' + a.scenario if a.synthetic else ''}"
    out = Path(a.out or root / "reports" / f"m6_{tag}")
    cache = a.cache_dir or ("/content/m6_cache" if Path("/content").is_dir() else None)
    run(out, Path(a.data_dir), Path(a.ckpt_dir or root / "checkpoints" / f"m6_{tag}"), cache and Path(cache), cfg,
        a.preset, a.device, a.synthetic, a.scenario, a.n_synthetic, a.stop_after_epochs, a.bootstrap, a.seed,
        a.workers)


if __name__ == "__main__":
    main()
