"""M12 metrics, heat maps and artery/vein segment voting (SYNTHETIC inputs, numpy only)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from drpinn.datasets.synthetic import fake_fundus_av
from drpinn.evaluation import (binary_report, classification_report, confusion, plot_confusion_heatmap, r2_score,
                               regression_report)
from drpinn.segmentation.av import av_label_map, masks_from_probabilities, vessel_pieces
from drpinn.segmentation.metrics import av_label_metrics, segmentation_metrics
from drpinn.vessel_graph.validation import compare_centrelines

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_binary_report_matches_hand_counts():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0], bool)
    p = np.array([1, 1, 0, 1, 0, 0, 0, 0], bool)
    r = binary_report(y, p)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 4)
    assert r["accuracy"] == pytest.approx(6 / 8)
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3) and r["specificity"] == pytest.approx(4 / 5)


def test_classification_report_per_class_and_macro():
    y = [0, 0, 1, 1, 2, 2]
    p = [0, 1, 1, 1, 2, 0]
    r = classification_report(y, p, ["a", "b", "c"])
    assert r["accuracy"] == pytest.approx(4 / 6)
    assert r["per_class"]["a"]["precision"] == pytest.approx(1 / 2)
    assert r["per_class"]["b"]["recall"] == pytest.approx(1.0)
    assert r["per_class"]["c"]["f1"] == pytest.approx(2 / 3)
    assert r["macro_f1"] == pytest.approx(np.mean([0.5, 0.8, 2 / 3]))
    assert np.array(r["confusion_matrix"]).sum() == 6


def test_confusion_rejects_out_of_range_labels():
    with pytest.raises(ValueError):
        confusion([0, 5], [0, 1], 5)


def test_r2_perfect_mean_and_worse_than_mean():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(t, t) == pytest.approx(1.0)
    assert r2_score(t, np.full(4, t.mean())) == pytest.approx(0.0)
    assert r2_score(t, t[::-1]) < 0
    assert np.isnan(r2_score([1.0, 1.0], [1.0, 2.0]))
    assert regression_report(t, t + 1)["mae"] == pytest.approx(1.0)


def test_segmentation_metrics_now_report_precision():
    ref = np.zeros((10, 10), bool)
    ref[2:5, 2:5] = True
    prob = ref.astype(float)
    prob[0, 0] = 1.0
    m = segmentation_metrics(prob, ref, np.ones_like(ref))
    assert m["precision"] == pytest.approx(9 / 10)


def test_compare_centrelines_returns_matched_radius_pairs():
    path = [np.array([[0.0, 0.0], [0.0, 50.0]])]
    c = compare_centrelines(path, [2.2], path, [2.0])
    assert c.radius_test.tolist() == [2.2] and c.radius_ref.tolist() == [2.0]


def test_confusion_heatmap_writes_png(tmp_path):
    out = plot_confusion_heatmap([[5, 1], [2, 7]], ["artery", "vein"], tmp_path / "cm.png", "x")
    assert out.exists() and out.stat().st_size > 1000


def _noisy_av(seed, noise=3.0):
    rgb, v, fov, a, w = fake_fundus_av(320, seed)
    rng = np.random.default_rng(seed)
    sig = np.where(a & ~w, 0.6, np.where(w & ~a, -0.6, 0.0))
    d = 3 * (sig + noise * ndimage.gaussian_filter(rng.normal(0, 1, v.shape), 2))
    prob = np.stack([v.astype(float), 0.9 / (1 + np.exp(-d)), 0.9 / (1 + np.exp(d))])
    ref = np.zeros(v.shape, np.uint8)
    ref[a & ~w], ref[w & ~a], ref[a & w] = 1, 2, 3
    return prob, v, ref


def test_vessel_pieces_cover_every_vessel_pixel():
    _, v, _ = _noisy_av(0)
    pieces, clusters, n = vessel_pieces(v)
    assert n > 5 and (pieces[v] > 0).all() and (pieces[~v] == 0).all() and clusters.max() > 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_segment_vote_beats_pixel_rules_on_noisy_maps(seed):
    prob, v, ref = _noisy_av(seed)
    scores = {m: av_label_metrics(av_label_map(prob, method=m, vessel=v), ref, v, "x")["x_accuracy"]
              for m in ("threshold", "argmax", "vote")}
    assert scores["vote"] > scores["threshold"] + 0.03
    assert scores["vote"] > scores["argmax"] + 0.03
    assert scores["vote"] > 0.93


def test_segment_vote_labels_every_vessel_pixel_and_nothing_else():
    prob, v, _ = _noisy_av(1)
    lab = av_label_map(prob, method="vote", vessel=v)
    assert (lab[v] > 0).all() and (lab[~v] == 0).all()


def test_masks_default_is_unchanged_original_rule():
    prob, v, _ = _noisy_av(0)
    vessel, a, w = masks_from_probabilities(prob)
    assert np.array_equal(a, vessel & (prob[1] >= 0.5)) and np.array_equal(w, vessel & (prob[2] >= 0.5))
    _, a2, w2 = masks_from_probabilities(prob, method="vote")
    assert ((a2 | w2) == vessel).all()


def test_av_label_map_rejects_unknown_method():
    with pytest.raises(ValueError):
        av_label_map(np.zeros((3, 4, 4)), method="nope")


def test_m12_script_end_to_end_on_synthetic(tmp_path):
    spec = importlib.util.spec_from_file_location("m12", SCRIPTS / "m12_evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    probs, drive, rite, pred = mod.make_synthetic(tmp_path / "in")
    res = mod.run(tmp_path / "out", probs, drive, rite, pred, error_maps=1, synthetic=True)
    seg, tri = res["segmentation"], res["triage"]
    assert seg["photos"] == 4 and 0.9 < seg["vessel_pooled"]["accuracy"] <= 1
    assert seg["artery_vein"]["vote"]["accuracy"] > seg["artery_vein"]["threshold"]["accuracy"]
    assert set(tri) == {"test_internal", "external_idrid_test"}
    assert np.isfinite(tri["test_internal"]["grade_r2"]["r2"])
    for name in ("heatmap_vessel_confusion.png", "heatmap_av_confusion_vote.png", "heatmap_seg_per_image.png",
                 "heatmap_triage_test_internal.png", "errors_1.png", "metrics.md"):
        assert (tmp_path / "out" / name).exists(), name
    assert json.loads((tmp_path / "out" / "metrics.json").read_text())["synthetic"] is True
