import math

import pytest
import torch

from core import Question, aggregate, build_report, calibrate, calibration_key, permutations_for, true_probs


def test_cyclic_permutations_cover_every_position_once():
    perms = permutations_for(5, 5, seed=0)
    assert len(perms) == 5
    for option in range(5):
        assert sorted(p.index(option) for p in perms) == list(range(5))


def test_extra_permutations_are_distinct():
    perms = permutations_for(4, 10, seed=1)
    assert len({tuple(p) for p in perms}) == 10


def test_permutations_capped_at_factorial():
    assert len(permutations_for(3, 50, seed=0)) == math.factorial(3)


def test_aggregations_are_distributions():
    vp = torch.tensor([[0.7, 0.2, 0.1], [0.5, 0.4, 0.1], [0.1, 0.8, 0.1]])
    for method in ("mean", "geomean", "weighted", "vote"):
        out = aggregate(vp, method)
        assert torch.isclose(out.sum(), torch.tensor(1.0))
    assert aggregate(vp, "vote").argmax() == 0


def test_geomean_penalises_options_rejected_by_one_variant():
    vp = torch.tensor([[0.6, 0.4], [0.6, 0.4], [0.001, 0.999]])
    assert aggregate(vp, "mean").argmax() == 1
    assert aggregate(vp, "geomean").argmax() == 1
    vp = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.01, 0.99]])
    assert aggregate(vp, "mean").argmax() == 0
    assert aggregate(vp, "geomean").argmax() == 1


def test_calibrate_quantile_matches_conformal_formula():
    probs = [i / 100 for i in range(1, 100)]
    cal = calibrate(probs, 0.1, "k")
    scores = sorted(1 - p for p in probs)
    assert cal.qhat == scores[math.ceil(100 * 0.9) - 1]
    assert cal.threshold == pytest.approx(1 - cal.qhat)


def test_calibrate_small_set_includes_everything():
    assert calibrate([0.9, 0.8], 0.1, "k").qhat == 1.0


def _variant(probs):
    return {"probs": probs, "letter_mass": 1.0, "order": list(range(len(probs)))}


def test_build_report_prediction_sets_and_coverage():
    qs = [Question("q1", ["a", "b"], "A"), Question("q2", ["a", "b"], "C")]
    variants = [[_variant([0.9, 0.05, 0.05])], [_variant([0.35, 0.2, 0.45])]]
    cal = calibrate([0.9] * 18 + [0.5, 0.3], 0.05, "k")
    assert cal.threshold == pytest.approx(0.3)
    rep = build_report(qs, variants, "mean", cal)
    r1, r2 = rep["results"]
    assert r1["prediction_set"] == ["A"] and r1["decided"]
    assert set(r2["prediction_set"]) == {"A", "C"} and not r2["decided"]
    assert rep["summary"]["conformal"]["coverage"] == 1.0
    assert rep["summary"]["accuracy"] == 1.0
    assert true_probs(rep) == pytest.approx([0.9, 0.45])


def test_calibration_key_ignores_aggregation_without_scramble():
    assert calibration_key("m", "it", 1, "vote") == calibration_key("m", "it", 1, "mean")
    assert calibration_key("m", "it", 5, "vote") != calibration_key("m", "it", 5, "mean")


def test_weighted_follows_variants_with_more_letter_mass():
    vp = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    assert aggregate(vp, "weighted", torch.tensor([0.1, 0.9])).argmax() == 1
    assert aggregate(vp, "weighted", torch.tensor([0.9, 0.1])).argmax() == 0


def test_all_options_per_type():
    assert Question("q", ["a", "b"]).all_options == ["a", "b", "Nessuna delle altre"]
    assert Question("q", ["a", "b"], none_option=False).all_options == ["a", "b"]
    assert Question("q", ["1", "2", "3"], type="score").all_options == ["1", "2", "3"]
    assert Question("q", type="yesno").all_options == ["Sì", "No"]


def test_validate_rejects_bad_answer_and_type():
    with pytest.raises(ValueError):
        Question("q", type="yesno", answer="C").validate()
    with pytest.raises(ValueError):
        Question("q", ["a", "b"], type="ranking").validate()
    with pytest.raises(ValueError):
        Question("q", [str(i) for i in range(11)], type="score").validate()


def test_score_is_probability_weighted_level():
    qs = [Question("urgenza", ["bassa", "media", "alta"], type="score"), Question("ok?", type="yesno")]
    rep = build_report(qs, [[_variant([0.2, 0.3, 0.5])], [_variant([0.8, 0.2])]], "mean")
    score, yesno = rep["results"]
    assert score["score"] == pytest.approx(0.2 * 1 + 0.3 * 2 + 0.5 * 3)
    assert score["score_norm"] == pytest.approx((score["score"] - 1) / 2)
    assert yesno["p_yes"] == pytest.approx(0.8)


def test_calibration_metrics():
    qs = [Question("q", ["a"], answer="A"), Question("q", ["a"], answer="B")]
    rep = build_report(qs, [[_variant([0.75, 0.25])], [_variant([0.75, 0.25])]], "mean")
    s = rep["summary"]
    assert s["ece"] == pytest.approx(0.25)
    assert s["brier"] == pytest.approx((0.25**2 * 2 + 0.75**2 * 2) / 2)
    assert s["nll"] == pytest.approx(-(math.log(0.75) + math.log(0.25)) / 2)


def test_first_permutation_is_original_order():
    for n in (2, 4, 5):
        assert permutations_for(n, 5, seed=3)[0] == list(range(n))


def test_multi_letter_answer_is_invalid():
    with pytest.raises(ValueError):
        Question("q", ["a", "b"], answer="AB").validate()
