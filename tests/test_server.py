import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import server
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
    monkeypatch.setattr(server, "ROOT", tmp_path)
    body = TestClient(server.app).get("/api/bench").json()
    assert body == {"eval": {"qwen3.5-2b": {"data": {"configs": {}}}}, "stress": {}}


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
