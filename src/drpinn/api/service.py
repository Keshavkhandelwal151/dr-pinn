"""Milestone 9: one photo in, every research output of the pipeline out.

Components, each optional; whatever is missing is reported with the reason
and the step that makes it, and the rest still runs:
  quality   always (relative to the APTOS training photos when the M6 results exist)
  triage    M6 model + its validation-fixed operating point (reports/m6_<preset>/m6_results.json)
  vessels   M5 model (checkpoints/m5_<preset>/final.pt): vessel, artery and vein maps and graphs
  shear     M7 Monte Carlo wall-shear-stress ranking per tree, aggregated to 32 px patches

The photo is processed exactly as in the milestones: triage on the <= 1024 px
copy cropped to the field of view (M6); vessels and physics at the working
scale where the field of view is 540 px across (M5, M7). The optic disc comes
from the user (preferred) or from a stated heuristic (disc.py).
"""

from __future__ import annotations

import base64
import io
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .. import DISCLAIMER, __version__
from ..regional.pipeline import M7Config, analyse_eye
from ..segmentation.preprocess import estimate_fov
from ..triage.data import limit_side, square_fundus, to_input
from ..triage.quality import QualityReference, quality_metrics
from ..uncertainty.montecarlo import MCConfig
from .disc import DISC_TO_FOV, disc_mask, estimate_disc

WORK_FOV_PX = 540


@dataclass(frozen=True)
class ServiceConfig:
    project: Path = Path(".")
    triage_preset: str = "full"
    seg_preset: str = "full"
    device: str | None = None
    mc_samples: int = 50
    max_upload_mb: float = 25.0
    av_method: str = "vote"  # artery/vein per vessel segment (M12); "threshold" reproduces M7 exactly

    @property
    def triage_ckpt(self) -> Path:
        return Path(self.project) / "checkpoints" / f"m6_{self.triage_preset}" / "best.pt"

    @property
    def triage_results(self) -> Path:
        return Path(self.project) / "reports" / f"m6_{self.triage_preset}" / "m6_results.json"

    @property
    def seg_ckpt(self) -> Path:
        return Path(self.project) / "checkpoints" / f"m5_{self.seg_preset}" / "final.pt"

    @property
    def seg_results(self) -> Path:
        return Path(self.project) / "reports" / f"m5_{self.seg_preset}" / "m5_results.json"


def radius_sigma_from_m5(results_path: Path) -> tuple[float, str]:
    """Same rule as the M7 plan: sigma = median |relative radius error| in M5 cross-validation / 0.674."""
    try:
        cv = json.loads(Path(results_path).read_text())["cv"]["summary"]["radius_median_abs_error"]["mean"]
        return round(float(cv) / 0.674, 3), f"M5 cross-validation (median radius error {cv:.3f})"
    except (OSError, KeyError, TypeError, ValueError):
        return 0.10, "plan default; no M5 cross-validation result found"


def png_b64(rgb: NDArray[np.uint8]) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, np.uint8)).save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _heat(rgb: NDArray[np.uint8], heat: NDArray, alpha: float = 0.45, mask: NDArray[np.bool_] | None = None):
    from matplotlib import colormaps

    col = colormaps["inferno"](np.clip(np.nan_to_num(heat), 0, 1))[..., :3] * 255
    a = alpha * (np.ones(heat.shape, bool) if mask is None else mask)[..., None]
    return (rgb * (1 - a) + col * a).astype(np.uint8)


def _ring(img: NDArray[np.uint8], centre_rc, diameter, colour=(0, 255, 0)) -> NDArray[np.uint8]:
    out = img.copy()
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    d = np.hypot(yy - centre_rc[0], xx - centre_rc[1])
    out[np.abs(d - diameter / 2) < max(1.0, diameter / 40)] = colour
    return out


def clean_json(x):
    """NaN/inf -> None and numpy -> Python, so the response is valid JSON."""
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.bool_):
        return bool(x)
    return x


class Analyser:
    """Loads whatever models exist under the project folder; thread-safe analyse()."""

    def __init__(self, config: ServiceConfig = ServiceConfig(), seg_predictor=None):
        import torch

        self.cfg = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        self.status: dict[str, dict] = {}
        self._seg_predictor = seg_predictor
        self._load_triage()
        self._load_segmentation()
        self.radius_sigma, self.radius_source = radius_sigma_from_m5(config.seg_results)

    # -- loading -------------------------------------------------------------------------
    def _load_triage(self) -> None:
        from ..triage.report import OperatingPoint
        from ..triage.train import load_triage

        self.triage = None
        self.qref = QualityReference.absolute_only()
        ck, res = self.cfg.triage_ckpt, self.cfg.triage_results
        if not ck.exists():
            self.status["triage"] = {"available": False, "detail": f"no trained model at {ck} (notebook step 15)"}
            return
        if not res.exists():
            self.status["triage"] = {"available": False, "detail": f"model not evaluated yet: {res} is missing. Run "
                                     "step 15, which fixes the temperature, threshold and uncertainty limit on validation."}
            return
        r = json.loads(res.read_text())
        if r.get("status") != "complete":
            self.status["triage"] = {"available": False, "detail": "the M6 evaluation did not finish (step 15)"}
            return
        model, ckd = load_triage(ck, self.device)
        self.triage = {"model": model, "op": OperatingPoint(**r["operating_point"]),
                       "size": int(ckd["config"]["image_size"]), "arch": ckd["config"]["arch"]}
        self.qref = QualityReference.from_dict(r["quality_reference"])
        self.status["triage"] = {"available": True, "detail": f"{ckd['config']['arch']}, epoch {ckd['epoch']}, "
                                 f"evaluated with drpinn {r.get('drpinn_version')}",
                                 "operating_point": r["operating_point"]}

    def _load_segmentation(self) -> None:
        if self._seg_predictor is not None:
            self.status["vessels"] = {"available": True, "detail": "injected predictor (tests)", "artery_vein": True}
            return
        ck = self.cfg.seg_ckpt
        if not ck.exists():
            self._seg_model = None
            self.status["vessels"] = {"available": False, "detail": f"no vessel model at {ck} (notebook step 12)"}
            return
        from ..segmentation.train import load_checkpoint

        self._seg_model, ckd = load_checkpoint(ck, self.device)
        av = True
        try:
            av = bool(json.loads(self.cfg.seg_results.read_text()).get("use_av", True))
        except (OSError, ValueError):
            pass
        self.status["vessels"] = {"available": True, "artery_vein": av,
                                  "detail": f"U-Net from {ck.name}" + ("" if av else " (no artery/vein training: "
                                                                        "shear risk needs RITE, step 11c)")}

    def _predict_vessels(self, rgb_w: NDArray[np.uint8], fov_w: NDArray[np.bool_]) -> NDArray[np.float32]:
        if self._seg_predictor is not None:
            return np.asarray(self._seg_predictor(rgb_w, fov_w), np.float32)
        from ..segmentation.infer import predict_image
        from ..segmentation.preprocess import preprocess

        return predict_image(self._seg_model, preprocess(rgb_w, fov_w), fov_w, tile=256, device=self.device)

    def health(self) -> dict:
        shear_ok = self.status["vessels"]["available"] and self.status["vessels"].get("artery_vein", False)
        return clean_json({"version": __version__, "device": self.device, "disclaimer": DISCLAIMER,
                           "components": {**self.status, "quality": {"available": True,
                                          "detail": "relative to APTOS training photos" if self.triage
                                          else "absolute rules only (no M6 results)"},
                                          "shear": {"available": shear_ok,
                                                    "detail": "Monte Carlo on artery and vein graphs" if shear_ok
                                                    else "needs the artery/vein model (step 12 with RITE)"}},
                           "radius_sigma": self.radius_sigma, "radius_sigma_source": self.radius_source,
                           "default_mc_samples": self.cfg.mc_samples})

    # -- analysis ------------------------------------------------------------------------
    def analyse(self, rgb: NDArray[np.uint8], disc_xy_d: tuple[float, float, float | None] | None = None,
                mc_samples: int | None = None, images: bool = True) -> dict:
        with self._lock:
            return clean_json(self._analyse(np.asarray(rgb)[..., :3].astype(np.uint8), disc_xy_d,
                                            mc_samples or self.cfg.mc_samples, images))

    def _analyse(self, rgb, disc_xy_d, mc_samples, images) -> dict:
        t0 = time.perf_counter()
        seconds = {}
        h, w = rgb.shape[:2]
        out = {"version": __version__, "disclaimer": DISCLAIMER, "image": {"width": w, "height": h}}

        small = limit_side(rgb)
        q = quality_metrics(small)
        flags = self.qref.flags(q)
        out["quality"] = {"metrics": q, "flags": flags,
                          "reference": "range of the APTOS training photos" if self.triage else "absolute rules only"}
        seconds["quality"] = time.perf_counter() - t0

        t = time.perf_counter()
        out["triage"] = self._triage(small, flags, images)
        seconds["triage"] = time.perf_counter() - t

        t = time.perf_counter()
        fov_small = estimate_fov(small)
        fov_d_full = 2 * np.sqrt(fov_small.sum() / np.pi) * (w / small.shape[1])
        if fov_small.sum() < 100:
            out["disc"] = {"available": False, "detail": "no field of view found"}
            out["vessels"] = out["shear"] = {"available": False, "detail": "no field of view found"}
            out["seconds"] = {**seconds, "total": time.perf_counter() - t0}
            return out
        scale = WORK_FOV_PX / fov_d_full
        rgb_w = _resize(rgb, scale)
        fov_w = estimate_fov(rgb_w)
        prob = self._predict_vessels(rgb_w, fov_w) if self.status["vessels"]["available"] else None
        seconds["vessels"] = time.perf_counter() - t

        if disc_xy_d is not None:
            x, y, d = disc_xy_d
            disc_rc = (y * scale, x * scale)
            disc_d = (d * scale) if d else DISC_TO_FOV * WORK_FOV_PX
            source = "given" + ("" if d else f" (diameter assumed {DISC_TO_FOV} x field of view)")
        else:
            disc_rc, disc_d = estimate_disc(rgb_w, fov_w, None if prob is None else prob[0])
            source = (f"estimated: brightest{' and most vessel-dense' if prob is not None else ''} region, "
                      f"diameter assumed {DISC_TO_FOV} x field of view. Click the disc to correct it.")
        out["disc"] = {"available": True, "source": source, "x": disc_rc[1] / scale, "y": disc_rc[0] / scale,
                       "diameter": disc_d / scale}

        if prob is None:
            out["vessels"] = {"available": False, "detail": self.status["vessels"]["detail"]}
            out["shear"] = {"available": False, "detail": "needs the vessel model"}
        else:
            out["vessels"], out["shear"] = self._vessels_and_shear(rgb_w, fov_w, prob, disc_rc, disc_d, scale,
                                                                   mc_samples, images)
        seconds["shear"] = time.perf_counter() - t - seconds["vessels"]
        out["seconds"] = {**{k: round(v, 2) for k, v in seconds.items()}, "total": round(time.perf_counter() - t0, 2)}
        return out

    def _triage(self, small, flags, images) -> dict:
        if self.triage is None:
            return {"available": False, "detail": self.status["triage"]["detail"]}
        from ..triage.model import grad_cam
        from ..triage.report import triage_card
        from ..triage.train import predict_logits

        size, model, op = self.triage["size"], self.triage["model"], self.triage["op"]
        sq = square_fundus(small, size)
        logits = predict_logits(model, sq[None], self.device)[0]
        card = triage_card(logits, op, flags)
        res = {"available": True, **card, "operating_point": op.to_dict(), "model": self.triage["arch"]}
        if images:
            cam = grad_cam(model, to_input(sq[None], self.device), "referable")[0]
            res["gradcam_png"] = png_b64(_heat(sq, cam, 0.45, sq.max(axis=2) > 10))
            res["input_png"] = png_b64(sq)
        return res

    def _vessels_and_shear(self, rgb_w, fov_w, prob, disc_rc, disc_d, scale, mc_samples, images):
        from ..segmentation.av import masks_from_probabilities

        cfg = M7Config(mc=MCConfig(n_samples=int(mc_samples), radius_sigma=self.radius_sigma),
                       av_method=self.cfg.av_method)
        vessel, artery, vein = masks_from_probabilities(prob, cfg.threshold, cfg.av_method)
        vessels = {"available": True, "working_scale": {"fov_px": WORK_FOV_PX, "scale": scale},
                   "vessel_fraction_of_fov": float(vessel[fov_w].mean()),
                   "artery_vein": self.status["vessels"].get("artery_vein", True)}
        if images:
            ov = rgb_w.copy()
            ov[vessel & fov_w] = (230, 230, 230)
            ov[artery & fov_w] = (220, 40, 40)
            ov[vein & fov_w] = (40, 90, 230)
            vessels["overlay_png"] = png_b64(_ring(ov, disc_rc, disc_d))
        if not vessels["artery_vein"]:
            return vessels, {"available": False, "detail": "the vessel model has no artery/vein training"}
        try:
            od = disc_mask(fov_w.shape, disc_rc, disc_d)
            rows, info = analyse_eye("upload", prob, fov_w, od, None, cfg)
        except ValueError as exc:
            return vessels, {"available": False, "detail": f"physics not possible on this photo: {exc}"}
        vessels["graphs"] = {k[6:]: v for k, v in info.items() if k.startswith("graph_")}
        p = cfg.patch_px
        ph, pw = int(np.ceil(fov_w.shape[0] / p)), int(np.ceil(fov_w.shape[1] / p))
        grid = {k: np.full((ph, pw), np.nan) for k in ("high_shear_mc", "low_shear_mc")}
        for r in rows:
            for k in grid:
                grid[k][r["row"], r["col"]] = r[k]
        order = sorted(rows, key=lambda r: -np.nan_to_num(r["high_shear_mc"], nan=-1))[:5]
        shear = {
            "available": True, "patch_px_working": p, "patch_px_original": p / scale, "rows": ph, "cols": pw,
            "high_shear": grid["high_shear_mc"], "low_shear": grid["low_shear_mc"], "patches_analysed": len(rows),
            "top_patches": [{"x": (r["col"] + 0.5) * p / scale, "y": (r["row"] + 0.5) * p / scale,
                             "p_top10": r["high_shear_mc"], "dist_disc_dd": r["dist_disc_dd"]} for r in order],
            "trees": {t: {"segments": info[f"{t}_segments"], "solved": info[f"{t}_segments_solved"],
                          "rank_stability": info[f"{t}_rank_stability"]} for t in ("artery", "vein")},
            "assumptions": _assumptions(cfg.mc, self.radius_source),
            "meaning": "Share of Monte Carlo samples in which a vessel segment in this patch was among the 10 % "
                       "highest wall-shear-stress segments of its own tree. A research ranking under assumed "
                       "boundary conditions, not a measurement and not a lesion prediction.",
        }
        if images:
            hi = np.kron(np.nan_to_num(grid["high_shear_mc"], nan=0.0), np.ones((p, p)))[:fov_w.shape[0], :fov_w.shape[1]]
            has = np.kron(np.isfinite(grid["high_shear_mc"]), np.ones((p, p), bool))[:fov_w.shape[0], :fov_w.shape[1]]
            shear["overlay_png"] = png_b64(_ring(_heat(rgb_w, hi, 0.55, has), disc_rc, disc_d, (0, 255, 0)))
        return vessels, shear


def _assumptions(mc: MCConfig, radius_source: str) -> list[str]:
    return [
        f"{mc.n_samples} Monte Carlo samples per tree",
        f"Vessel ends: terminal resistance of k ~ U{tuple(mc.levels)} unseen Murray generations, "
        f"outlet pressure jitter ±{mc.outlet_jitter_mmhg} mmHg",
        f"Per-segment radius error: lognormal, sigma {mc.radius_sigma} ({radius_source})",
        f"Viscosity: Newtonian or Fåhræus–Lindqvist (Pries 1992, Hd ~ U{tuple(mc.hd_range)}), probability {mc.p_pries}",
        f"Pixel size from an assumed optic-disc diameter ~ U{tuple(mc.disc_um_range)} µm",
        "In each vessel piece the end nearest the optic disc is the disc end; pieces with no other end are left out",
        "Absolute pressures are illustrative; only rankings within one tree are reported",
    ]


def _resize(rgb: NDArray[np.uint8], scale: float) -> NDArray[np.uint8]:
    from PIL import Image

    h, w = rgb.shape[:2]
    size = (max(32, round(w * scale)), max(32, round(h * scale)))
    return np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).resize(size, Image.LANCZOS))
