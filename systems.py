"""Decision models next to our LLMs: Clef-flash, d3-lite, GLiDE, vela3.1-fast and Laya on the local GPU, Jev and
Gemini Decision 1 through their APIs. They answer the same typed questions with one probability per option, so they run as a model with one pass
and the server, the calibration and the UI treat them like our LLMs."""
import gc
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "bench"))

from core import DEFAULT_BATCH_ROWS, DEFAULT_BATCH_TOKENS, DEFAULT_PROMPT, MAX_QUESTIONS, MODELS, Engine  # noqa: E402
import compare  # noqa: E402


@dataclass(frozen=True)
class DecisionSpec:
    key: str
    vendor: str
    where: str
    env: str | None = None
    module: str | None = None
    kind: str = "decision"

    @property
    def label(self) -> str:
        return compare.BASELINES[self.key]["label"]

    @property
    def repo(self) -> str:
        return compare.BASELINES[self.key]["repo"]

    @property
    def params_b(self) -> float | None:
        return compare.BASELINES[self.key]["params_b"]


DECISION_MODELS = [
    DecisionSpec("clef", "Cloudflare", "gpu"),
    DecisionSpec("d3", "vLLM-SR", "gpu"),
    DecisionSpec("glide", "Fastino", "gpu"),
    DecisionSpec("vela", "vLLM-SR", "gpu"),
    DecisionSpec("laya", "Convai", "gpu", module="laya"),
    DecisionSpec("jev", "TypeSafe", "api", env="CLASSIFIER_DEV_API"),
    DecisionSpec("gemini", "Google", "api", env="GOOGLE_API_KEY"),
]
DECISION_BY_KEY = {d.key: d for d in DECISION_MODELS}
RUNNERS = {"clef": compare.ClefRunner, "laya": compare.LayaRunner, "jev": compare.JevRunner,
           "gemini": compare.GeminiRunner, "d3": compare.D3Runner, "glide": compare.GlideRunner,
           "vela": compare.VelaRunner}
LLM_VENDORS = {"qwen": "Alibaba", "lfm": "Liquid AI", "granite": "IBM"}


def unavailable(spec: DecisionSpec) -> str | None:
    """Why a decision model cannot run here, or None."""
    if spec.module and importlib.util.find_spec(spec.module) is None:
        return "Serve il gruppo baselines: uv sync --group baselines"
    if spec.env:
        try:
            compare.env_key(spec.env)
        except SystemExit:
            return f"Manca {spec.env} nel file .env"
    return None


def catalog() -> list[dict]:
    """Every selectable model, our LLMs first: the fields the UI needs to group and describe them."""
    llms = [{**asdict(m), "kind": "llm", "where": "gpu",
             "vendor": next(v for k, v in LLM_VENDORS.items() if m.key.startswith(k))} for m in MODELS]
    decision = [{"key": d.key, "label": d.label, "repo": d.repo, "params_b": d.params_b, "vendor": d.vendor,
                 "kind": d.kind, "where": d.where, "set_attention": False, "unavailable": unavailable(d)}
                for d in DECISION_MODELS]
    return llms + decision


class DecisionEngine:
    """Same interface as core.Engine for the server: one pass per question, no permutations or prompt styles."""

    def __init__(self, persist: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.persist = persist
        self.spec = self.runner = None
        self.load_seconds = 0.0

    def unload(self):
        self.spec = self.runner = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, key: str) -> dict:
        if self.spec and self.spec.key == key:
            return self.info()
        spec = DECISION_BY_KEY[key]
        if reason := unavailable(spec):
            raise RuntimeError(reason)
        self.unload()
        t0 = time.perf_counter()
        runner = RUNNERS[key]
        self.runner = runner(self.device, persist=self.persist) if spec.where == "api" else runner(self.device)
        self.load_seconds = time.perf_counter() - t0
        self.spec = spec
        return self.info()

    def info(self) -> dict:
        if not self.spec:
            return {"loaded": None}
        mem = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        return {"loaded": self.spec.key, "label": self.spec.label, "repo": self.spec.repo, "kind": "decision",
                "load_seconds": round(self.load_seconds, 2), "vram_gb": round(mem, 2),
                "device": "api" if self.spec.where == "api" else self.device}

    def run_variants(self, questions: list, scramble: int = 1, batch_size: int = DEFAULT_BATCH_ROWS,
                     prompt_style: str = DEFAULT_PROMPT, seed: int = 0, max_questions: int | None = MAX_QUESTIONS,
                     prefix_cache: bool = False, set_attention: bool = False,
                     batch_tokens: int = DEFAULT_BATCH_TOKENS) -> tuple[list[list[dict]], dict]:
        if self.runner is None:
            raise RuntimeError("Nessun modello caricato")
        if not questions:
            raise ValueError("Nessuna domanda")
        if max_questions and len(questions) > max_questions:
            raise ValueError(f"Massimo {max_questions} domande per richiesta")
        for q in questions:
            q.validate()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        probs = self.runner([compare.item(q) for q in questions])
        elapsed = time.perf_counter() - t0
        variants = [[{"order": list(range(len(p))), "position_probs": p, "probs": p, "letter_mass": 1.0,
                      "unconstrained_top": None}] for p in probs]
        width = max(len(p) for p in probs)
        bias = [sum(p[j] for p in probs if j < len(p)) / max(1, sum(j < len(p) for p in probs)) for j in range(width)]
        timing = {"prompts": len(questions), "total_s": elapsed, "tokenize_s": 0.0, "forward_s": elapsed,
                  "prompts_per_s": len(questions) / elapsed if elapsed else None, "tokens": 0, "tokens_per_s": None,
                  "padding_ratio": 0.0, "batches": 1, "max_batch_rows": len(questions), "oom_retries": 0,
                  "prefix_mode": "off", "tokens_saved": 0,
                  "peak_vram_gb": (torch.cuda.max_memory_allocated() / 2**30
                                   if torch.cuda.is_available() and self.spec.where == "gpu" else 0.0),
                  "position_bias": bias,
                  "sample_prompt": json.dumps(compare.system_one_request(questions[0]), ensure_ascii=False, indent=2)}
        return variants, timing

    score = Engine.score


class Engines:
    """Keeps one model in memory: loading a decision model unloads the LLM and the other way round."""

    def __init__(self, persist: bool = False):
        self.llm, self.decision = Engine(), DecisionEngine(persist)
        self.active = self.llm

    def get(self, key: str):
        engine = self.decision if key in DECISION_BY_KEY else self.llm
        if engine is not self.active:
            self.active.unload()
            self.active = engine
        return engine

    def load(self, key: str) -> dict:
        return self.get(key).load(key)

    def info(self) -> dict:
        return self.active.info()
