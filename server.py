import argparse
import json
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core import (
    AGGREGATIONS, DEFAULT_BATCH_ROWS, DEFAULT_BATCH_TOKENS, DEFAULT_PROMPT, LETTERS, MAX_QUESTIONS, MODELS, MODELS_BY_KEY, NONE_OPTION, PROMPT_STYLES,
    PROMPT_STYLES_BY_KEY, Calibration, Engine, Question, build_report, calibrate, calibration_key,
    true_probs,
)

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
CALIBRATIONS = ROOT / "data" / "calibrations.json"
sys.path.insert(0, str(ROOT / "bench"))

app = FastAPI(title="DecisionLLM")
engine = Engine()
gpu_lock = threading.Lock()


class QuestionIn(BaseModel):
    question: str = Field(min_length=1)
    options: list[str] = Field(default_factory=list, max_length=len(LETTERS))
    answer: str | None = None
    type: Literal["choice", "score", "yesno"] = "choice"
    none_option: bool = True
    state: str | None = None
    id: str | None = None


class RunIn(BaseModel):
    model: str
    state: str | None = None
    questions: list[QuestionIn] = Field(default_factory=list, max_length=MAX_QUESTIONS)
    scramble: int = Field(1, ge=1, le=24)
    aggregation: str = "geomean"
    batch_size: int = Field(DEFAULT_BATCH_ROWS, ge=1, le=1024)
    batch_tokens: int = Field(DEFAULT_BATCH_TOKENS, ge=64, le=131072)
    prompt_style: str = DEFAULT_PROMPT
    prefix_cache: bool = True
    set_attention: bool = False


class CalibrateIn(RunIn):
    epsilon: float = Field(0.1, gt=0, lt=1)
    source: str = "questions"


def load_calibrations() -> dict[str, Calibration]:
    if not CALIBRATIONS.exists():
        return {}
    return {k: Calibration(**v) for k, v in json.loads(CALIBRATIONS.read_text()).items()}


def calibration_out(cal: Calibration) -> dict:
    return {**asdict(cal), "threshold": cal.threshold}


def store_calibration(cal: Calibration):
    cals = load_calibrations()
    cals[cal.key] = cal
    CALIBRATIONS.parent.mkdir(exist_ok=True)
    CALIBRATIONS.write_text(json.dumps({k: asdict(v) for k, v in cals.items()}, indent=1))


YES_NO_ANSWERS = {"SÌ": "A", "SI": "A", "YES": "A", "NO": "B"}


def to_questions(items: list[QuestionIn], state: str | None = None) -> list[Question]:
    out = []
    for q in items:
        answer = (q.answer or "").strip().upper() or None
        if q.type == "yesno" and answer:
            answer = YES_NO_ANSWERS.get(answer, answer)
        options = [o.strip() for o in q.options]
        if not all(options):
            raise HTTPException(422, f"Opzione vuota in {q.question!r}: le lettere delle risposte si sposterebbero")
        question = Question(q.question.strip(), options, answer, q.type, q.none_option, q.state or state, q.id)
        try:
            question.validate()
        except ValueError as err:
            raise HTTPException(422, str(err))
        out.append(question)
    return out


def validate(req: RunIn):
    if req.model not in MODELS_BY_KEY:
        raise HTTPException(422, f"Modello sconosciuto: {req.model}")
    if req.aggregation not in AGGREGATIONS:
        raise HTTPException(422, f"Aggregazione sconosciuta: {req.aggregation}")
    if req.prompt_style not in PROMPT_STYLES_BY_KEY:
        raise HTTPException(422, f"Prompt sconosciuto: {req.prompt_style}")
    if req.set_attention and not MODELS_BY_KEY[req.model].set_attention:
        raise HTTPException(422, f"{MODELS_BY_KEY[req.model].label} non supporta il set-based attention")


def key_for(req: RunIn) -> str:
    return calibration_key(req.model, req.prompt_style, req.scramble, req.aggregation, req.set_attention)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config():
    gpu = None
    if torch.cuda.is_available():
        gpu = {"name": torch.cuda.get_device_name(0),
               "total_gb": torch.cuda.get_device_properties(0).total_memory / 2**30}
    return {
        "models": [asdict(m) for m in MODELS],
        "prompt_styles": [asdict(p) for p in PROMPT_STYLES],
        "aggregations": list(AGGREGATIONS),
        "default_prompt": DEFAULT_PROMPT,
        "default_batch_tokens": DEFAULT_BATCH_TOKENS,
        "default_batch_rows": DEFAULT_BATCH_ROWS,
        "max_questions": MAX_QUESTIONS,
        "none_option": NONE_OPTION,
        "gpu": gpu,
        "engine": engine.info(),
        "calibrations": {k: calibration_out(v) for k, v in load_calibrations().items()},
    }


@app.post("/api/load")
def load(req: RunIn):
    validate(req)
    with gpu_lock:
        return engine.load(req.model)


@app.get("/api/examples")
def examples():
    return json.loads((ROOT / "data" / "examples.json").read_text())


@app.get("/api/test-questions")
def test_questions():
    return json.loads((ROOT / "data" / "test_questions.json").read_text())


@app.get("/api/bench")
def bench():
    """Benchmark summaries (eval and stress) from bench/results, for the comparison page."""
    results = ROOT / "bench" / "results"
    out = {"eval": {}, "stress": {}}
    for phase in out:
        for path in sorted(results.glob(f"{phase}-*.json")):
            out[phase][path.stem.removeprefix(f"{phase}-")] = json.loads(path.read_text())
    return out


@app.post("/api/score")
def score(req: RunIn):
    validate(req)
    questions = to_questions(req.questions, req.state)
    if not questions:
        raise HTTPException(422, "Nessuna domanda")
    cal = load_calibrations().get(key_for(req))
    with gpu_lock:
        engine.load(req.model)
        try:
            return engine.score(questions, req.scramble, req.aggregation, req.batch_size, req.prompt_style, cal,
                                prefix_cache=req.prefix_cache, set_attention=req.set_attention,
                                batch_tokens=req.batch_tokens)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(507, "Memoria GPU esaurita anche con batch size 1")


@app.post("/api/calibrate")
def calibrate_endpoint(req: CalibrateIn):
    validate(req)
    if req.source == "global-mmlu-dev":
        from run_bench import load_split
        questions = [i["question"] for i in load_split("dev")]
    else:
        questions = [q for q in to_questions(req.questions, req.state) if q.answer]
    if len(questions) < 10:
        raise HTTPException(422, "Servono almeno 10 domande con risposta nota per calibrare")
    with gpu_lock:
        engine.load(req.model)
        variants, timing = engine.run_variants(questions, req.scramble, req.batch_size, req.prompt_style,
                                               max_questions=None, prefix_cache=req.prefix_cache,
                                               set_attention=req.set_attention, batch_tokens=req.batch_tokens)
    method = req.aggregation if req.scramble > 1 else "mean"
    cal = calibrate(true_probs(build_report(questions, variants, method)), req.epsilon, key_for(req))
    store_calibration(cal)
    report = build_report(questions, variants, method, cal)
    return {"calibration": calibration_out(cal), "summary": report["summary"], "forward_s": timing["forward_s"]}


class DecisionIn(BaseModel):
    type: Literal["choice", "score", "yesno"]
    instructions: str = Field(min_length=1)
    criteria: dict[str, str] | list[str] = Field(default_factory=list)
    allow_none: bool = False


class DecideIn(BaseModel):
    model: str
    state: str | None = None
    questions: dict[str, DecisionIn] = Field(min_length=1, max_length=MAX_QUESTIONS)
    scramble: int = Field(1, ge=1, le=24)
    aggregation: str = "geomean"
    prompt_style: str = DEFAULT_PROMPT
    prefix_cache: bool = True
    set_attention: bool = False


def typed_answer(spec: DecisionIn, item: dict) -> dict:
    keys = list(spec.criteria) if isinstance(spec.criteria, dict) else list(spec.criteria)
    if spec.type == "yesno":
        keys = ["yes", "no"]
    elif spec.type == "choice" and spec.allow_none:
        keys.append("none")
    probs = dict(zip(keys, item["probs"]))
    top = keys[LETTERS.index(item["argmax"])]
    out = {"probabilities": probs, "confidence": item["confidence"]}
    if spec.type == "choice":
        out["choice"] = top
    elif spec.type == "score":
        out.update(score=item["score"], score_norm=item["score_norm"], level=top)
    else:
        out["p_yes"] = item["p_yes"]
    if "prediction_set" in item:
        out["prediction_set"] = [keys[LETTERS.index(letter)] for letter in item["prediction_set"]]
        out["decided"] = item["decided"]
    return out


@app.post("/api/decide")
def decide(req: DecideIn):
    """Typed decisions on a shared state: one forward pass per question (per permutation)."""
    run = RunIn(model=req.model, scramble=req.scramble, aggregation=req.aggregation, prompt_style=req.prompt_style,
                set_attention=req.set_attention)
    validate(run)
    questions = []
    for key, spec in req.questions.items():
        options = list(spec.criteria.values()) if isinstance(spec.criteria, dict) else list(spec.criteria)
        q = Question(spec.instructions, options, None, spec.type, spec.allow_none, req.state, key)
        try:
            q.validate()
        except ValueError as err:
            raise HTTPException(422, f"{key}: {err}")
        questions.append(q)
    with gpu_lock:
        engine.load(req.model)
        rep = engine.score(questions, req.scramble, req.aggregation, prompt_style=req.prompt_style,
                           calibration=load_calibrations().get(key_for(run)), prefix_cache=req.prefix_cache,
                           set_attention=req.set_attention)
    s = rep["summary"]
    return {
        "model": rep["model"]["repo"],
        "answers": {spec_key: typed_answer(spec, item)
                    for (spec_key, spec), item in zip(req.questions.items(), rep["results"])},
        "timing": {"total_ms": 1000 * s["total_s"], "forward_ms": 1000 * s["forward_s"], "prompts": s["prompts"],
                   "tokens": s["tokens"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--preload", choices=list(MODELS_BY_KEY))
    args = ap.parse_args()
    if args.preload:
        engine.load(args.preload)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
