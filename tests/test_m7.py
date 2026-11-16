"""M7: Monte Carlo risk, patch aggregation, statistics, and null/positive controls (SYNTHETIC)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from drpinn.datasets.synthetic import fake_fundus_av
from drpinn.regional.patches import estimate_fovea, segment_patches
from drpinn.regional.pipeline import M7Config, analyse_eye, disc_from_mask
from drpinn.regional.stats import Logistic, average_precision, evaluate_score, paired_difference, per_image_auroc, t_interval
from drpinn.segmentation.av import av_graphs
from drpinn.uncertainty.montecarlo import MCConfig, disc_ends, tree_risk
from drpinn.vessel_graph.extract import VesselGraph, extract_vessel_graph
from drpinn.vessel_graph.synthetic_masks import make_image, y_bifurcation

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m7_lesion_association.py"
NO_UNCERTAINTY = MCConfig(n_samples=4, radius_sigma=0.0, p_pries=0.0, outlet_jitter_mmhg=0.0, levels=(6.0, 6.0))


def _eye(seed=3, size=576):
    img, v, fov, a, w, disc = fake_fundus_av(size, seed, return_disc=True)
    yy, xx = np.mgrid[:size, :size]
    od = (yy - disc[0]) ** 2 + (xx - disc[1]) ** 2 <= (0.06 * size) ** 2
    return img, v, fov, a, w, disc, od


# --- Monte Carlo ------------------------------------------------------------------

def test_disc_end_is_the_tip_nearest_the_disc():
    g = extract_vessel_graph(y_bifurcation().binary)
    trunk_start = np.array([100.0, 30.0])  # y_bifurcation trunk starts at column 30, row 100
    inlets, outlets, _ = disc_ends(g, tuple(trunk_start))
    assert len(inlets) == 1 and len(outlets) == 2
    assert np.linalg.norm(g.node_rc[inlets[0]] - trunk_start) < 5


def test_without_uncertainty_the_ranking_is_certain():
    _, v, fov, a, w, disc, od = _eye()
    g = av_graphs(v, a, w, fov=fov)
    r = tree_risk(g.artery, disc, 0.12 * 576, "artery", NO_UNCERTAINTY)
    vals = set(np.round(r.p_high[r.solved], 6))
    assert vals <= {0.0, 1.0} and r.rank_stability == pytest.approx(1.0)
    assert np.nansum(r.p_high) == pytest.approx(max(1, round(0.1 * r.solved.sum())), abs=1)


def _chain_graph(radii):
    """Hand-built graph: segments in series along a line (one flow path)."""
    k = len(radii)
    nodes = np.array([[100.0, 20.0 + 60.0 * i] for i in range(k + 1)])
    edges = np.array([[i, i + 1] for i in range(k)])
    paths = tuple(np.vstack([nodes[i], nodes[i + 1]]) for i in range(k))
    kinds = ("tip",) + ("junction",) * (k - 1) + ("tip",)
    return VesselGraph((200, 400), nodes, kinds, np.zeros(k + 1, bool), np.zeros(k + 1, bool), edges, paths,
                       np.full(k, 60.0), np.asarray(radii, float), np.asarray(radii, float), np.ones(k, bool),
                       np.zeros(k, bool), {})


def test_narrowed_segment_in_series_is_always_the_high_shear_one():
    """Same flow through every segment: WSS ~ 1/r^3, so a 50 % narrowing is ~8x - robust to all sampled unknowns."""
    g = _chain_graph([5.0, 5.0, 2.5, 5.0, 5.0])
    r = tree_risk(g, (100.0, 20.0), 80.0, "artery", MCConfig(n_samples=30, top_fraction=0.2))
    assert r.p_high[2] == 1.0 and np.all(r.p_high[[0, 1, 3, 4]] == 0.0)
    assert r.p_low[2] == 0.0


def test_pieces_without_an_outflow_end_are_left_out_not_fatal():
    """A chain (solvable) + a closed loop + a loop with a single end: the last two cannot carry flow."""
    nodes = np.array([[100.0, 20.0], [100.0, 80.0], [100.0, 140.0],                        # chain 0-1-2
                      [30.0, 300.0], [60.0, 330.0], [30.0, 360.0],                          # loop 3-4-5
                      [150.0, 260.0], [150.0, 300.0], [180.0, 330.0], [150.0, 360.0]])      # end 6, loop 7-8-9
    edges = np.array([[0, 1], [1, 2], [3, 4], [4, 5], [5, 3], [6, 7], [7, 8], [8, 9], [9, 7]])
    k = len(edges)
    paths = tuple(np.vstack([nodes[u], nodes[v]]) for u, v in edges)
    kinds = ("tip", "junction", "tip", "junction", "junction", "junction", "tip", "junction", "junction", "junction")
    r = np.full(k, 4.0)
    g = VesselGraph((200, 400), nodes, kinds, np.zeros(10, bool), np.zeros(10, bool), edges, paths,
                    np.full(k, 50.0), r, r, np.ones(k, bool), np.zeros(k, bool), {})
    res = tree_risk(g, (100.0, 20.0), 80.0, "artery", MCConfig(n_samples=5))
    assert res.solved.tolist() == [True, True] + [False] * 7
    assert np.all(np.isfinite(res.p_high[:2])) and np.all(np.isnan(res.p_high[2:]))


def test_tree_risk_input_checks():
    _, v, fov, a, w, disc, od = _eye()
    g = av_graphs(v, a, w, fov=fov)
    assert tree_risk(None, disc, 50.0, "artery") is None
    with pytest.raises(ValueError):
        tree_risk(g.artery, disc, 0.0, "artery")
    with pytest.raises(ValueError):
        tree_risk(g.artery, disc, 50.0, "capillary")


def test_mask_loader_reads_every_common_encoding(tmp_path):
    """IDRiD-style TIF masks come in several encodings; all must give the same binary mask."""
    from PIL import Image

    from drpinn.datasets.idrid import describe_image_file, load_binary_mask

    m = np.zeros((40, 60), bool)
    m[10:25, 20:45] = True
    red = np.zeros((40, 60, 3), np.uint8)
    red[m] = (255, 0, 0)
    rgba = np.dstack([red, np.full((40, 60), 255, np.uint8)])
    pal = Image.fromarray(m.astype(np.uint8), "L").convert("P")
    pal.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    images = {"one_bit": Image.fromarray(m).convert("1"), "grey_01": Image.fromarray(m.astype(np.uint8)),
              "grey_255": Image.fromarray(m.astype(np.uint8) * 255), "red_rgb": Image.fromarray(red),
              "red_rgba": Image.fromarray(rgba, "RGBA"), "palette": pal,
              "grey_16bit": Image.fromarray(m.astype(np.uint16) * 65535)}
    for name, im in images.items():
        path = tmp_path / f"{name}.tif"
        im.save(path)
        np.testing.assert_array_equal(load_binary_mask(path), m, err_msg=name)
    empty = tmp_path / "empty.tif"
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(empty)
    assert not load_binary_mask(empty).any() and "values 0-0" in describe_image_file(empty)


# --- patches ----------------------------------------------------------------------

def test_segment_patch_mapping_and_pooling():
    img = make_image((64, 64), [np.array([[5.0, 2.0], [5.0, 60.0]])], [2.0], "horizontal")
    g = extract_vessel_graph(img.binary)
    sp = segment_patches(g, 16, (64, 64))
    assert sorted(sp[0].tolist()) == [0, 1, 2, 3]  # top row of a 4 x 4 patch grid


def test_patch_score_is_the_maximum_over_its_segments():
    from drpinn.regional.patches import _patch_max
    seg_patches = [np.array([0, 1]), np.array([1, 2]), np.array([2])]
    out = _patch_max(4, seg_patches, np.array([0.9, 0.2, np.nan]))
    np.testing.assert_array_equal(out[:3], [0.9, 0.9, 0.2])
    assert out[3] == -np.inf


def test_patch_labels_and_disc_estimate():
    od = np.zeros((100, 200), bool)
    od[40:60, 20:40] = True
    (r, c), d = disc_from_mask(od)
    assert (r, c) == pytest.approx((49.5, 29.5)) and d == pytest.approx(2 * np.sqrt(400 / np.pi))
    fov = np.ones((100, 200), bool)
    assert estimate_fovea((r, c), d, fov)[1] > c  # fovea lies towards the image centre


def test_analyse_eye_rows_have_scores_labels_and_no_lesion_leak():
    img, v, fov, a, w, disc, od = _eye()
    les = {k: np.zeros_like(fov) for k in ("MA", "HE", "EX", "SE")}
    les["HE"][300:306, 300:306] = True
    prob = np.stack([v, a, w]).astype(float)
    rows, info = analyse_eye("e", prob, fov, od, les, M7Config(mc=MCConfig(n_samples=5)))
    assert rows and {"high_shear_mc", "red_lesion", "vessel_density", "near_fovea"} <= set(rows[0])
    rows2, _ = analyse_eye("e", prob, fov, od, None, M7Config(mc=MCConfig(n_samples=5)))
    # scores must not depend on where lesions are
    for r1, r2 in zip(rows, rows2):
        assert r1["patch"] == r2["patch"] and r1["high_shear_mc"] == pytest.approx(r2["high_shear_mc"], nan_ok=True)
    assert 0 <= info["red_lesion_labelled_vessel"] <= 1


# --- statistics -------------------------------------------------------------------

def test_average_precision_hand_example():
    assert average_precision([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0]) == pytest.approx((1 + 2 / 3) / 2)


def test_per_image_auroc_and_t_interval():
    s = np.array([0.9, 0.1, 0.2, 0.8, 0.5, 0.5])
    y = np.array([1, 0, 0, 1, 1, 0], bool)
    g = np.array([0, 0, 1, 1, 2, 2])
    per = per_image_auroc(np.r_[s, 0.3, 0.4], np.r_[y, False, False], np.r_[g, 3, 3])
    assert per == {0: 1.0, 1: 1.0, 2: 0.5}  # image 3 has no lesion patch: not scored
    m, lo, hi = t_interval([0.6, 0.7, 0.8])
    assert m == pytest.approx(0.7) and lo < 0.7 < hi
    assert np.isnan(t_interval([0.6])[1])


def test_paired_difference_detects_better_score():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(20), 30)
    y = rng.random(g.size) < 0.2
    good = y + rng.normal(0, 0.5, g.size)
    noise = rng.normal(0, 1, g.size)
    d = paired_difference(good, noise, y, g)
    assert d["difference"] > 0.2 and d["ci"][0] > 0 and d["p_value"] < 0.01
    e = evaluate_score(noise, y, g, n_boot=200)
    assert e["within_ci"][0] < 0.5 < e["within_ci"][1]


def test_logistic_regression_separates_and_standardises_on_train_only():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 2))
    y = (x[:, 0] + 0.2 * rng.normal(size=400)) > 0
    m = Logistic().fit(x * [1000, 1], y)
    pred = m.decision(x * [1000, 1]) > 0
    assert (pred == y).mean() > 0.9 and m.w[1] > 0


# --- controls (the whole script) ---------------------------------------------------

_MOD = {}


def _m7():
    if "m" not in _MOD:  # load once, so the cached synthetic eyes are shared between the two controls
        spec = importlib.util.spec_from_file_location("m7", SCRIPT)
        _MOD["m"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MOD["m"])
    return _MOD["m"]


def _run(scenario, tmp_path):
    return _m7().run(tmp_path, synthetic=True, scenario=scenario, n_synthetic=15,
                  cfg=M7Config(mc=MCConfig(n_samples=10)), n_boot=100, figures=1)


def test_positive_control_is_detected(tmp_path):
    s = _run("planted", tmp_path)
    p = s["primary"]
    assert p["verdict"] == "positive association" and p["within_image_auroc"] > 0.8
    assert (tmp_path / "m7_forest.png").exists() and (tmp_path / "m7_patches.csv").exists()
    assert json.loads((tmp_path / "m7_results.json").read_text())["source"].startswith("SYNTHETIC")


def test_null_control_finds_no_association(tmp_path):
    s = _run("null", tmp_path)
    p = s["primary"]
    assert p["ci"][0] <= 0.5 <= p["ci"][1]
    assert p["verdict"].startswith("no evidence")
