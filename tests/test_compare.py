import math

import pytest

import compare
from core import LETTERS


def test_temperature_one_keeps_probabilities():
    p = [0.7, 0.2, 0.1]
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(compare.apply_temperature(p, 1.0), p))


def test_high_temperature_flattens_distribution():
    flat = compare.apply_temperature([0.9, 0.05, 0.05], 20.0)
    assert max(flat) - min(flat) < 0.1


def test_fit_temperature_softens_overconfident_predictions():
    probs = [[0.99, 0.01]] * 6 + [[0.01, 0.99]] * 4
    answers = ["A"] * 6 + ["A"] * 4
    assert compare.fit_temperature(probs, answers) > 1


def test_fit_temperature_sharpens_underconfident_predictions():
    probs = [[0.55, 0.45]] * 10
    assert compare.fit_temperature(probs, ["A"] * 10) < 1


def test_evaluate_reports_temperature_metrics_with_dev_split():
    dev = [compare.item(compare.Question("q", ["x", "y"], "A", none_option=False)) for _ in range(20)]
    test = [compare.item(compare.Question("q", ["x", "y"], a, none_option=False)) for a in "AABA"]
    probs = lambda items: [[0.8, 0.2] for _ in items]  # noqa: E731
    out = compare.evaluate(dev, test, probs(dev), probs(test))
    assert out["accuracy"] == 0.75
    assert {"ece_t", "brier_t", "nll_t", "temperature", "conformal"} <= out.keys()


def test_evaluate_without_dev_split_skips_calibration():
    test = [compare.item(compare.Question("q", ["x", "y"], "A", none_option=False))]
    out = compare.evaluate([], test, [], [[0.6, 0.4]])
    assert out["accuracy"] == 1 and "conformal" not in out


def test_massive_options_contain_the_correct_intent_once():
    for it in compare.massive_split("test")[:50]:
        q = it["question"]
        assert len(q.all_options) == compare.MASSIVE_OPTIONS
        assert len(set(q.options)) == len(q.options)
        assert q.answer in LETTERS[:compare.MASSIVE_OPTIONS]


def test_sib_yesno_is_balanced():
    items = compare.sib_yesno_split("test")
    yes = sum(it["question"].answer == "A" for it in items)
    assert abs(yes - len(items) / 2) <= 1


def test_laya_probabilities_follow_option_order():
    q = compare.Question("q", ["Basso", "Alto"], type="score")
    assert compare.LayaRunner.probs(q, {"probabilities": {"0": 0.3, "1": 0.7}}) == [0.3, 0.7]
    q = compare.Question("q", type="yesno")
    assert compare.LayaRunner.probs(q, {"noul": 0.8}) == pytest.approx([0.8, 0.2])


def test_nli_duplicate_options_give_a_distribution():
    runner = object.__new__(compare.NliRunner)
    runner.pipe = lambda text, labels, **kw: {"labels": labels, "scores": [0.4, 0.3, 0.2, 0.1][:len(labels)]}
    q = compare.Question("q", ["Roma", "Roma", "Milano"], "A")
    probs = runner([compare.item(q)], "local")[0]
    assert len(probs) == len(q.all_options)
    assert sum(probs) == pytest.approx(1.0)


def test_results_are_saved_per_dataset_and_resumed(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "RESULTS", tmp_path)
    monkeypatch.setattr(compare, "save", lambda name, data: (tmp_path / f"{name}.json").write_text(compare.json.dumps(data)))
    compare.store("laya", {"label": "Laya"}, "massive", 1)
    compare.store("laya", {"label": "Laya"}, "local", 2)
    saved = compare.json.loads((tmp_path / "compare-laya.json").read_text())
    assert saved == {"system": {"label": "Laya"}, "datasets": {"massive": 1, "local": 2}}
    assert compare.pending("laya", ["massive", "local", "mmlu"], fresh=False) == ["mmlu"]
    assert compare.pending("laya", ["massive", "mmlu"], fresh=True) == ["massive", "mmlu"]
    assert not (tmp_path / "compare-laya.json").exists()


def test_jev_runner_maps_answers_and_reuses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "JEV_CACHE", tmp_path / "cache.json")
    monkeypatch.setattr(compare, "JEV_PAUSE_S", 0)
    monkeypatch.setenv("CLASSIFIER_DEV_API", "test")
    calls = []

    def fake_request(self, body):
        calls.append(body)
        return {"model": "jev-1.13.0", "answers": {"q": {"type": "choice", "probabilities": {"A": 0.0, "B": 0.95, "C": 0.05}}}}, 200.0

    monkeypatch.setattr(compare.JevRunner, "request", fake_request)
    q = compare.Question("q", ["x", "y"], "B", state="testo")
    items = [compare.item(q)]
    runner = compare.JevRunner("cpu")
    assert runner(items)[0] == pytest.approx([0.0, 0.95, 0.05])
    assert compare.JevRunner("cpu")(items) and len(calls) == 1
    assert runner.ms(items) == pytest.approx(0.2) and runner.model == "jev-1.13.0"


def test_retry_after_accepts_seconds_and_http_dates():
    assert compare.retry_after("3", 1) == 3
    assert compare.retry_after("Wed, 21 Oct 2015 07:28:00 GMT", 1) == 0
    assert compare.retry_after(None, 4) == 4
    assert compare.retry_after("soon", 2) == 2


def test_missing_env_file_gives_a_clear_message(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "ROOT", tmp_path)
    monkeypatch.delenv("CLASSIFIER_DEV_API", raising=False)
    with pytest.raises(SystemExit, match="CLASSIFIER_DEV_API"):
        compare.env_key("CLASSIFIER_DEV_API")
