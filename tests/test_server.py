import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import server
import systems
from server import QuestionIn, to_questions


def test_blank_option_is_rejected_instead_of_shifting_the_answer():
    with pytest.raises(HTTPException) as err:
        to_questions([QuestionIn(question="Capitale?", options=["Roma", "Parigi", "", "Madrid"], answer="B")])
    assert err.value.status_code == 422


def test_multi_letter_answer_is_rejected():
    with pytest.raises(HTTPException):
        to_questions([QuestionIn(question="Capitale?", options=["Roma", "Parigi"], answer="ab")])


def test_yes_no_answers_are_mapped_to_letters():
    q = to_questions([QuestionIn(question="Ok?", type="yesno", answer="sì")])[0]
    assert q.answer == "A"


def test_bench_endpoint_returns_results_by_phase_and_model(tmp_path, monkeypatch):
    results = tmp_path / "bench" / "results"
    results.mkdir(parents=True)
    (results / "eval-qwen3.5-2b.json").write_text(json.dumps({"data": {"configs": {}}}))
    (results / "raw-qwen3.5-2b.json").write_text("{}")
    (results / "compare-laya.json").write_text(json.dumps({"datasets": {}}))
    monkeypatch.setattr(server, "ROOT", tmp_path)
    body = TestClient(server.app).get("/api/bench").json()
    assert body == {"eval": {"qwen3.5-2b": {"data": {"configs": {}}}}, "stress": {},
                    "compare": {"laya": {"datasets": {}}}}


def test_typed_answer_reports_prediction_set_with_criteria_keys():
    spec = server.DecisionIn(type="yesno", instructions="Rimborso?")
    item = {"probs": [0.67, 0.33], "argmax": "A", "confidence": 0.67, "p_yes": 0.67,
            "prediction_set": ["A", "B"], "decided": False}
    out = server.typed_answer(spec, item)
    assert out["prediction_set"] == ["yes", "no"] and out["decided"] is False


def test_config_exposes_calibration_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CALIBRATIONS", tmp_path / "calibrations.json")
    server.store_calibration(server.Calibration(epsilon=0.1, qhat=0.9, n=20, key="k"))
    assert server.config()["calibrations"]["k"]["threshold"] == pytest.approx(0.1)


class FakeRunner:
    def __init__(self, device, persist=True):
        self.persist = persist

    def __call__(self, items):
        return [[0.7, 0.3] if it["question"].type == "yesno" else [0.6, 0.3, 0.1] for it in items]


@pytest.fixture
def fake_gemini(monkeypatch):
    monkeypatch.setitem(systems.RUNNERS, "gemini", FakeRunner)
    monkeypatch.setattr(systems, "unavailable", lambda spec: None)
    monkeypatch.setattr(server, "engine", systems.Engines())


def test_config_lists_decision_models_after_our_llms():
    models = server.config()["models"]
    kinds = [m["kind"] for m in models]
    assert kinds == sorted(kinds, key=["llm", "decision"].index)
    gemini = next(m for m in models if m["key"] == "gemini")
    assert gemini["where"] == "api" and "unavailable" in gemini


def test_missing_api_key_makes_a_decision_model_unavailable(monkeypatch):
    def no_key(name):
        raise SystemExit(f"{name} mancante")

    monkeypatch.setattr(systems.compare, "env_key", no_key)
    assert systems.unavailable(systems.DECISION_BY_KEY["gemini"]) == "Manca GOOGLE_API_KEY nel file .env"


def test_decision_model_scores_in_one_pass_and_ignores_permutations(fake_gemini):
    body = {"model": "gemini", "scramble": 5, "set_attention": True, "state": "Lampione spento.",
            "questions": [{"question": "Ufficio?", "options": ["Lavori", "Tributi"], "type": "choice"},
                          {"question": "Intervento?", "type": "yesno", "answer": "sì"}]}
    r = TestClient(server.app).post("/api/score", json=body).json()
    assert r["params"]["scramble"] == 1 and r["params"]["set_attention"] is False
    assert r["summary"]["prompts"] == 2 and r["model"]["device"] == "api"
    assert r["results"][0]["argmax"] == "A" and r["results"][1]["correct"] is True


def test_unavailable_decision_model_is_a_422(monkeypatch):
    monkeypatch.setattr(systems, "unavailable", lambda spec: "Manca GOOGLE_API_KEY nel file .env")
    monkeypatch.setattr(server, "engine", systems.Engines())
    r = TestClient(server.app).post("/api/load", json={"model": "gemini"})
    assert r.status_code == 422 and "GOOGLE_API_KEY" in r.json()["detail"]
