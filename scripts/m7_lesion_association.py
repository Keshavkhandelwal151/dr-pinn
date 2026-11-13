"""M7: are model-estimated high- (or low-) shear vessel regions where expert-marked lesions are?

Pre-registered analysis (docs/m7_analysis_plan.md - fixed before any real-data run):
  PRIMARY   On the 27 IDRiD official TEST images, the mean within-image AUROC of
            the Monte Carlo high-shear score (P(segment in its tree's top 10 % WSS),
            max over the patch) for red lesions (microaneurysms + haemorrhages),
            among vessel-bearing patches. 95 % t-interval over images.
            Compared with 0.5 and, paired, with the vessel-density and
            distance-to-fovea baselines.
  SECONDARY Low-shear score; deterministic scores (no uncertainty); geometry
            baselines; exudates and all lesions; all 81 images.
  EXPLORATORY Logistic models fitted on the 54 TRAIN images, scored on TEST:
            position + density, + geometry, + physics.
This is CROSS-SECTIONAL association with lesions that already exist - proxy
validation, not prediction of future lesions (PDF section 11.5).

    python scripts/m7_lesion_association.py --idrid-dir data/IDRiD --model checkpoints/m5_full/final.pt
    python scripts/m7_lesion_association.py --synthetic --scenario null      # software check only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drpinn import __version__  # noqa: E402
from drpinn.regional.pipeline import M7Config, analyse_eye  # noqa: E402
from drpinn.regional.stats import Logistic, evaluate_score, paired_difference  # noqa: E402
from drpinn.uncertainty.montecarlo import MCConfig  # noqa: E402

PRIMARY_SCORE, PRIMARY_LABEL = "high_shear_mc", "red_lesion"
SCORES = ["high_shear_mc", "low_shear_mc", "high_shear_deterministic", "low_shear_deterministic",
          "tortuosity", "thinness", "murray_deviation", "vessel_density", "near_fovea", "near_disc"]
LABELS = ["red_lesion", "MA", "HE", "exudate", "any_lesion"]
PAIRED_BASELINES = ["vessel_density", "near_fovea", "high_shear_deterministic", "thinness"]
FEATURE_SETS = {"position + density": ["vessel_density", "near_fovea", "near_disc"],
                "+ geometry": ["vessel_density", "near_fovea", "near_disc", "tortuosity", "thinness", "murray_deviation"],
                "+ geometry + physics": ["vessel_density", "near_fovea", "near_disc", "tortuosity", "thinness",
                                         "murray_deviation", "high_shear_mc", "low_shear_mc"]}


# ----------------------------------------------------------------------------- data

def _resize(arr: np.ndarray, size: tuple[int, int], kind: str) -> np.ndarray:
    from PIL import Image

    w, h = size
    if kind == "rgb":
        return np.asarray(Image.fromarray(arr).resize((w, h), Image.LANCZOS))
    # masks: a working pixel is set if ANY full-resolution pixel under it is set
    return np.asarray(Image.fromarray(arr.astype(np.float32)).resize((w, h), Image.BOX)) > 0


def load_mask(path) -> np.ndarray:
    """Binary mask in any common encoding (1-bit, 0/1 or 0/255 grey, palette, colour)."""
    from drpinn.datasets.idrid import load_binary_mask

    return load_binary_mask(path)


def idrid_eyes(idrid_dir: Path, cfg: M7Config):
    """Yield (image_id, split, rgb, fov, od_mask, lesions) at working scale for IDRiD segmentation images."""
    from PIL import Image

    from drpinn.datasets.idrid import find_idrid
    from drpinn.segmentation.preprocess import estimate_fov

    idx = find_idrid(idrid_dir)
    if not idx.segmentation:
        raise FileNotFoundError(f"no IDRiD segmentation images under {idrid_dir} (notebook step 11a)")
    for rec in idx.segmentation:
        masks = rec.extra["masks"]
        if "OD" not in masks:
            print(f"  {rec.image_id}: skipped (no optic-disc mask)", flush=True)
            continue
        with Image.open(rec.path) as im:
            rgb = np.asarray(im.convert("RGB"))
        small = np.asarray(Image.fromarray(rgb).resize((rgb.shape[1] // 8, rgb.shape[0] // 8)))
        fov_small = estimate_fov(small)
        diam_full = 2 * np.sqrt(fov_small.sum() / np.pi) * 8
        scale = cfg.fov_diameter_px / diam_full
        size = (max(32, round(rgb.shape[1] * scale)), max(32, round(rgb.shape[0] * scale)))
        rgb_w = _resize(rgb, size, "rgb")
        od_full = load_mask(masks["OD"])
        if not od_full.any():
            from drpinn.datasets.idrid import describe_image_file
            print(f"  {rec.image_id}: skipped (optic-disc mask is empty; {describe_image_file(masks['OD'])})", flush=True)
            continue
        od = _resize(od_full, size, "mask")
        lesions = {k: _resize(load_mask(p), size, "mask") for k, p in masks.items() if k != "OD"}
        for k in ("MA", "HE", "EX", "SE"):
            lesions.setdefault(k, np.zeros(od.shape, bool))  # absent mask = lesion type not annotated
        yield rec.image_id, rec.official_split or "train", rgb_w, estimate_fov(rgb_w), od, lesions


@lru_cache(maxsize=64)
def _fake_eye(size: int, seed: int):
    from drpinn.datasets.synthetic import fake_fundus_av
    return fake_fundus_av(size, seed, return_disc=True)


def synthetic_eyes(n: int, scenario: str, cfg: M7Config, seed: int = 0):
    """FAKE eyes with ground-truth artery/vein maps used as 'probabilities' (no model).

    scenario 'null': lesions in random vessel-bearing patches (no real association).
    scenario 'planted': lesions placed in the patches with the highest deterministic
    shear (a real association the analysis should find).
    """
    rng = np.random.default_rng(seed)
    size = 576
    for k in range(n):
        img, v, fov, a, w, disc = _fake_eye(size, 7000 + k)
        yy, xx = np.mgrid[:size, :size]
        od = (yy - disc[0]) ** 2 + (xx - disc[1]) ** 2 <= (0.06 * size) ** 2
        prob = np.stack([v, a, w]).astype(np.float32)
        rows, _ = analyse_eye(f"syn{k}", prob, fov, od, None, replace(cfg, mc=replace(cfg.mc, n_samples=1)))
        lesions = {t: np.zeros_like(fov) for t in ("MA", "HE", "EX", "SE")}
        if rows:
            order = (np.argsort([-r["high_shear_deterministic"] if np.isfinite(r["high_shear_deterministic"]) else 1
                                 for r in rows]) if scenario == "planted" else rng.permutation(len(rows)))
            for i in order[: max(2, len(rows) // 8)]:
                r0, c0 = rows[i]["row"] * cfg.patch_px, rows[i]["col"] * cfg.patch_px
                rr, cc = r0 + rng.integers(4, cfg.patch_px - 4), c0 + rng.integers(4, cfg.patch_px - 4)
                lesions["MA" if rng.random() < 0.5 else "HE"][rr - 2:rr + 3, cc - 2:cc + 3] = True
        yield f"syn{k}", ("test" if k % 3 == 0 else "train"), img, fov, od, lesions, prob


# ----------------------------------------------------------------------------- model

def predict_cached(model, device, image_id, rgb, fov, cache_dir: Path):
    from drpinn.segmentation.infer import predict_image
    from drpinn.segmentation.preprocess import preprocess

    path = cache_dir / f"{image_id}.npz"
    if path.exists():
        with np.load(path) as z:
            if z["prob"].shape[1:] == fov.shape:
                return z["prob"].astype(np.float32)
    prob = predict_image(model, preprocess(rgb, fov), fov, tile=256, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, prob=prob.astype(np.float16))
    return prob


# ----------------------------------------------------------------------------- analysis

def analyse_rows(rows: list[dict], n_boot: int, seed: int) -> dict:
    split = {r["image_id"]: r["split"] for r in rows}
    out = {"single_scores": {}, "paired_vs_baselines": {}, "combined_models": {}}
    for subset in ("test", "all"):
        sel = [r for r in rows if subset == "all" or r["split"] == "test"]
        if not sel:
            continue
        g = np.array([r["image_id"] for r in sel])
        for lab in LABELS:
            if lab not in sel[0]:
                continue
            y = np.array([r[lab] for r in sel], bool)
            for sc in SCORES:
                s = np.array([r[sc] for r in sel], float)
                out["single_scores"].setdefault(subset, {}).setdefault(lab, {})[sc] = evaluate_score(s, y, g, n_boot, seed)
        if PRIMARY_LABEL in sel[0]:
            y = np.array([r[PRIMARY_LABEL] for r in sel], bool)
            prim = np.array([r[PRIMARY_SCORE] for r in sel], float)
            for b in PAIRED_BASELINES:
                out["paired_vs_baselines"].setdefault(subset, {})[b] = paired_difference(
                    prim, np.array([r[b] for r in sel], float), y, g, n_boot, seed)
    train = [r for r in rows if split[r["image_id"]] != "test"]
    test = [r for r in rows if split[r["image_id"]] == "test"]
    if train and test and PRIMARY_LABEL in rows[0]:
        y_tr = np.array([r[PRIMARY_LABEL] for r in train], float)
        y_te = np.array([r[PRIMARY_LABEL] for r in test], bool)
        g_te = np.array([r["image_id"] for r in test])
        decisions = {}
        if 0 < y_tr.sum() < y_tr.size:
            for name, cols in FEATURE_SETS.items():
                m = Logistic().fit(np.array([[r[c] for c in cols] for r in train], float), y_tr)
                d = m.decision(np.array([[r[c] for c in cols] for r in test], float))
                decisions[name] = d
                out["combined_models"][name] = evaluate_score(d, y_te, g_te, n_boot, seed)
            out["combined_models"]["physics added (paired)"] = paired_difference(
                decisions["+ geometry + physics"], decisions["+ geometry"], y_te, g_te, n_boot, seed)
    return out


def headline(results: dict) -> dict:
    s = results.get("single_scores", {}).get("test", {}).get(PRIMARY_LABEL, {}).get(PRIMARY_SCORE)
    if not s:
        return {"available": False}
    lo, hi = s["within_ci"]
    verdict = ("no evidence of association (CI includes 0.5)" if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0.5 <= hi else
               "positive association" if lo > 0.5 else "negative association (lesions where shear is NOT high)")
    return {"available": True, "score": PRIMARY_SCORE, "label": PRIMARY_LABEL, "images": s["images_scored"],
            "within_image_auroc": s["within_image_auroc"], "ci": s["within_ci"], "p_vs_0.5": s["p_vs_0.5"], "verdict": verdict,
            "vs_baselines": results.get("paired_vs_baselines", {}).get("test", {})}


def run(out_dir: Path, idrid_dir: Path | None = None, model_path: Path | None = None, synthetic: bool = False,
        scenario: str = "null", n_synthetic: int = 12, cfg: M7Config = M7Config(), n_boot: int = 2000,
        seed: int = 0, max_images: int | None = None, figures: int = 2) -> dict:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, infos, model_meta, examples = [], [], {}, []
    if synthetic:
        eyes = ((i, s, rgb, fov, od, les, prob) for i, s, rgb, fov, od, les, prob in synthetic_eyes(n_synthetic, scenario, cfg, seed))
        source = f"SYNTHETIC ({scenario}) - software check, not results"
    else:
        import torch

        from drpinn.segmentation.train import load_checkpoint
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, ck = load_checkpoint(model_path, device)
        model_meta = {"checkpoint": str(model_path), "trained_on": ck.get("meta", {}).get("train_ids"),
                      "drpinn_version_at_training": ck.get("meta", {}).get("drpinn_version")}
        if not any(v > 0 for v in [ck["config"]["av_loss_weight"]]):
            print("warning: model has no artery/vein training", flush=True)

        def gen():
            for i, s, rgb, fov, od, les in idrid_eyes(idrid_dir, cfg):
                yield i, s, rgb, fov, od, les, predict_cached(model, device, i, rgb, fov, out_dir / "probs")
        eyes = gen()
        source = f"IDRiD segmentation set, model {model_path}"
    for k, (image_id, split, rgb, fov, od, les, prob) in enumerate(eyes):
        if max_images and k >= max_images:
            break
        t1 = time.perf_counter()
        try:
            r, info = analyse_eye(image_id, prob, fov, od, les, cfg)
        except ValueError as exc:
            print(f"  {image_id}: skipped ({exc})", flush=True)
            continue
        for row in r:
            row["split"] = split
        rows += r
        infos.append({**info, "split": split, "seconds": time.perf_counter() - t1})
        if len(examples) < figures:
            examples.append((image_id, rgb, fov, od, les, prob))
        print(f"  {image_id} ({split}): {len(r)} vessel patches, red-lesion patches {sum(x.get('red_lesion', 0) for x in r)}, "
              f"rank stability artery {info['artery_rank_stability']:.2f} / vein {info['vein_rank_stability']:.2f}", flush=True)
    if not rows:
        raise RuntimeError("no eye could be analysed")
    results = analyse_rows(rows, n_boot, seed)
    stab = {t: float(np.nanmedian([i[f"{t}_rank_stability"] for i in infos])) for t in ("artery", "vein")}
    contamination = [i["red_lesion_labelled_vessel"] for i in infos if np.isfinite(i.get("red_lesion_labelled_vessel", np.nan))]
    summary = {"source": source, "drpinn_version": __version__, "config": cfg.to_dict(), "model": model_meta,
               "images": len(infos), "test_images": sum(i["split"] == "test" for i in infos), "patches": len(rows),
               "primary": headline(results), "median_rank_stability": stab,
               "red_lesion_pixels_labelled_vessel_median": float(np.median(contamination)) if contamination else None,
               "results": results, "runtime_s": time.perf_counter() - t0}
    (out_dir / "m7_results.json").write_text(json.dumps(summary, indent=2, default=float))
    for name, table in (("m7_patches.csv", rows), ("m7_eyes.csv", infos)):
        with open(out_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for t in table for k in t)))
            w.writeheader()
            w.writerows(table)
    plot_forest(results, out_dir / "m7_forest.png", source)
    for ex in examples:
        plot_example(*ex, cfg, out_dir / f"m7_example_{ex[0]}.png")
    return summary


# ----------------------------------------------------------------------------- figures

def plot_forest(results: dict, path: Path, source: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = results.get("single_scores", {}).get("test", {}).get(PRIMARY_LABEL)
    if not s:
        return
    names = [n for n in SCORES if n in s]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.5))
    for k, n in enumerate(names[::-1]):
        e = s[n]
        col = "#d95f02" if n == PRIMARY_SCORE else ("#7570b3" if "shear" in n else "#666666")
        ax.plot(e["within_ci"], [k, k], color=col, lw=2)
        ax.plot(e["within_image_auroc"], k, "o", color=col)
    ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.set_yticks(range(len(names)), names[::-1], fontsize=9)
    ax.set_xlabel("mean within-image AUROC for red lesions (test images), 95 % t-interval over images")
    ax.set_title(f"Which scores point to lesion patches?  ({source})", fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_example(image_id, rgb, fov, od, lesions, prob, cfg: M7Config, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from drpinn.segmentation.av import av_graphs, masks_from_probabilities
    from drpinn.uncertainty.montecarlo import tree_risk
    from drpinn.regional.pipeline import disc_from_mask

    disc_rc, disc_px = disc_from_mask(od)
    v, a, w = masks_from_probabilities(prob, cfg.threshold, cfg.av_method)
    g = av_graphs(v & fov, a & fov, w & fov, fov=fov, vessel_prob=prob[0])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    axes[0].imshow(rgb)
    axes[0].set_title(f"{image_id}: photo, red lesions outlined", fontsize=9)
    red = np.zeros(fov.shape, bool)
    for k in ("MA", "HE"):
        red |= np.asarray(lesions.get(k, np.zeros_like(fov)), bool)
    for ax in axes:
        if red.any():
            ax.contour(red, levels=[0.5], colors="yellow", linewidths=0.8)
        ax.set_axis_off()
    axes[1].imshow(rgb * 0.35 / 255 + 0.0)
    lcs = []
    for name, graph, style in (("artery", g.artery, "-"), ("vein", g.vein, "--")):
        r = tree_risk(graph, disc_rc, disc_px, name, cfg.mc)
        if r is None:
            continue
        segs = [p[:, ::-1] for p in graph.paths_rc]
        lc = LineCollection(segs, array=np.nan_to_num(r.p_high), cmap="inferno", clim=(0, 1), linewidths=2.2,
                            linestyles=style)
        axes[1].add_collection(lc)
        lcs.append(lc)
    axes[1].set_title("P(top-10 % wall shear stress): artery solid, vein dashed", fontsize=9)
    if lcs:
        fig.colorbar(lcs[0], ax=axes[1], shrink=0.75, label="probability over Monte Carlo samples")
    plt.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--idrid-dir", default=str(root / "data" / "IDRiD"))
    ap.add_argument("--model", default=str(root / "checkpoints" / "m5_full" / "final.pt"))
    ap.add_argument("--out", default=str(root / "reports" / "m7"))
    ap.add_argument("--samples", type=int, default=50, help="Monte Carlo samples per tree")
    ap.add_argument("--radius-sigma", type=float, default=0.10, help="per-segment radius error (use the M5 CV value)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--scenario", choices=["null", "planted"], default="null")
    ap.add_argument("--n-synthetic", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--av-method", choices=["threshold", "argmax", "vote"], default="threshold",
                    help="artery/vein assignment; 'threshold' = pre-registered rule, 'vote' = segment vote (M12)")
    a = ap.parse_args()
    cfg = M7Config(mc=MCConfig(n_samples=a.samples, radius_sigma=a.radius_sigma, seed=a.seed), av_method=a.av_method)
    s = run(Path(a.out), Path(a.idrid_dir), Path(a.model), a.synthetic, a.scenario, n_synthetic=a.n_synthetic,
            cfg=cfg, n_boot=a.bootstrap, seed=a.seed, max_images=a.max_images)
    p = s["primary"]
    print(f"\n{s['source']}: {s['images']} eyes ({s['test_images']} test), {s['patches']} vessel patches")
    if p["available"]:
        print(f"PRIMARY: within-image AUROC {p['within_image_auroc']:.3f} "
              f"[{p['ci'][0]:.3f}, {p['ci'][1]:.3f}], p = {p['p_vs_0.5']:.3f} vs 0.5, over {p['images']} test images -> {p['verdict']}")
        for b, d in p["vs_baselines"].items():
            print(f"  vs {b:24s} difference {d['difference']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]  p={d['p_value']:.3f}")
    print(f"median rank stability: artery {s['median_rank_stability']['artery']:.2f}, vein {s['median_rank_stability']['vein']:.2f}")


if __name__ == "__main__":
    main()
