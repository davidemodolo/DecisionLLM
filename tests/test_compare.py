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
        return {"model": "jev-1.13.0", "answers": {"q": {"type": "choice", "probabilities": {"A": 0.0, "B": 0.95, "C": 0.05}}}}, 200.0, None

    monkeypatch.setattr(compare.JevRunner, "request", fake_request)
    q = compare.Question("q", ["x", "y"], "B", state="testo")
    items = [compare.item(q)]
    runner = compare.JevRunner("cpu")
    assert runner(items)[0] == pytest.approx([0.0, 0.95, 0.05])
    assert compare.JevRunner("cpu")(items) and len(calls) == 1
    assert runner.ms(items) == pytest.approx(0.2) and runner.model == "jev-1.13.0"


def test_gemini_runner_sends_typed_questions_and_answers_blocked_input_uniformly(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "GEMINI_CACHE", tmp_path / "gemini.json")
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    bodies = []

    def fake_request(self, body):
        bodies.append(body)
        if body["input"] == "bloccato":
            return None, 90.0, "safety"
        return {"answers": {"q": {"type": "score", "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}}}, 180.0, None

    monkeypatch.setattr(compare.GeminiRunner, "request", fake_request)
    score = compare.Question("Tono?", ["Neg", "Neu", "Pos"], "Pos", type="score", state="ottimo")
    blocked = compare.Question("q", ["x", "y"], "A", state="bloccato")
    items = [compare.item(score), compare.item(blocked)]
    runner = compare.GeminiRunner("cpu")
    out = runner(items)
    assert out[0] == pytest.approx([0.1, 0.2, 0.7])
    assert out[1] == pytest.approx([1 / len(blocked.all_options)] * len(blocked.all_options))
    assert bodies[0] == {"model": "gemini-decision-1", "input": "ottimo", "questions": {"q": {
        "type": "score", "instructions": "Tono?", "criteria": ["Neg", "Neu", "Pos"]}}}
    assert runner.model == "gemini-decision-1" and runner.blocked_count(items) == 1
    assert compare.GeminiRunner("cpu")(items) and len(bodies) == 2


def test_gemini_blocked_recognises_only_safety_errors(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    runner = compare.GeminiRunner.__new__(compare.GeminiRunner)
    safety = compare.json.dumps({"error": {"code": "safety", "message": "blocked"}}).encode()
    invalid = compare.json.dumps({"error": {"code": 400, "message": "criteria must have 2 options"}}).encode()
    assert runner.blocked(400, safety) == "blocked"
    assert runner.blocked(400, invalid) is None


def test_rate_limiter_waits_for_the_window(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(compare.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(compare.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    limiter = compare.RateLimiter(rpm=2, tpm=100)
    limiter.wait(40); limiter.add(40)
    limiter.wait(40); limiter.add(40)
    assert clock[0] == 0
    limiter.wait(10)
    assert clock[0] >= 60
    limiter.add(90)
    limiter.wait(20)
    assert clock[0] >= 120


def test_clef_answer_maps_option_ids_to_our_order():
    yesno = compare.Question("q", ["Sì", "No"], "A", type="yesno", state="s")
    choice = compare.Question("q", ["x", "y"], "B", state="s")
    p = {"true": 0.8, "false": 0.2}
    assert compare.LayaRunner.probs(yesno, compare.ClefRunner.answer(yesno, p)) == pytest.approx([0.8, 0.2])
    p = {k: 1 / len(choice.all_options) for k in compare.LETTERS[:len(choice.all_options)]}
    assert len(compare.LayaRunner.probs(choice, compare.ClefRunner.answer(choice, p))) == len(choice.all_options)


def test_cpu_rows_gathers_on_cpu():
    rows = compare.CpuRows(compare.torch.arange(12.0).reshape(4, 3), "cpu")
    assert rows[compare.torch.tensor([3, 1])].tolist() == [[9, 10, 11], [3, 4, 5]]


def test_retry_delay_reads_google_error_details():
    payload = compare.json.dumps({"error": {"details": [{"@type": "x"}, {"retryDelay": "23s"}]}}).encode()
    assert compare.retry_delay(payload, 1) == 24
    assert compare.retry_delay(b"not json", 5) == 5


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


def test_vela_turns_score_levels_into_a_choice_keyed_by_level():
    q = {"type": "score", "instructions": "Tono?", "criteria": ["Neg", "Neu", "Pos"]}
    assert compare.VelaRunner.as_choice(q) == {"type": "choice", "instructions": "Tono?",
                                               "criteria": {"0": "Neg", "1": "Neu", "2": "Pos"}}
    yesno = {"type": "noul", "instructions": "Ok?"}
    assert compare.VelaRunner.as_choice(yesno) is yesno


def test_system_one_runner_sends_jev_requests_and_maps_answers():
    class Fake(compare.SystemOneRunner):
        def answers(self, requests):
            self.sent = requests
            return [{"probabilities": {"0": 0.2, "1": 0.3, "2": 0.5}}, {"noul": 0.9}]

    score = compare.Question("Tono?", ["Neg", "Neu", "Pos"], "C", type="score", state="ottimo")
    yesno = compare.Question("Parla di sport?", ["Sì", "No"], "A", type="yesno", state="partita")
    runner = Fake()
    assert runner([compare.item(score), compare.item(yesno)]) == [[0.2, 0.3, 0.5], [0.9, pytest.approx(0.1)]]
    assert runner.sent[0] == {"state": "ottimo", "questions": {"q": {"type": "score", "instructions": "Tono?",
                                                                    "criteria": ["Neg", "Neu", "Pos"]}}}
