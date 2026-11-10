import copy
import gc
import math
import random
import sys
import time
from dataclasses import dataclass, field

import torch

if not torch.cuda.is_available():
    # flash-linear-attention kernels are Triton/CUDA only: hide them so Qwen3.5 uses the PyTorch path on CPU
    sys.modules["fla"] = None

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

MAX_QUESTIONS = 256
NONE_OPTION = "Nessuna delle altre"
YES_NO = ["Sì", "No"]
QUESTION_TYPES = ("choice", "score", "yesno")
MAX_SCORE_LEVELS = 10
LETTERS = "ABCDEFGH"
AGGREGATIONS = ("geomean", "mean", "weighted", "vote")


@dataclass(frozen=True)
class PromptStyle:
    key: str
    label: str
    instruction: str
    prefix: str
    conj: str = "or"
    context_label: str = "Context"

    def render(self, n_options: int) -> str:
        letters = list(LETTERS[:n_options])
        return self.instruction.format(letters=f"{', '.join(letters[:-1])} {self.conj} {letters[-1]}")


IT_STRICT = "Quale opzione è corretta? Rispondi con una sola lettera ({letters}), senza altro testo né formattazione."
EN_STRICT = "Which option is correct? Answer with a single letter ({letters}), with no other text or formatting."
PROMPT_STYLES = [
    PromptStyle("en-knowno", "English KnowNo + \"Answer:\"",
                "Which option is correct? Answer with a single letter.", "Answer:"),
    PromptStyle("en-strict", "English strict + \"Answer:\"", EN_STRICT, "Answer:"),
    PromptStyle("en-option", "English strict + \"The correct option is\"", EN_STRICT, "The correct option is"),
    PromptStyle("it-strict", "Italiano strict + \"Risposta:\"", IT_STRICT, "Risposta:", "o", "Contesto"),
    PromptStyle("it-option", "Italiano strict + \"L'opzione corretta è\"", IT_STRICT, "L'opzione corretta è", "o",
                "Contesto"),
]
PROMPT_STYLES_BY_KEY = {p.key: p for p in PROMPT_STYLES}
DEFAULT_PROMPT = "it-strict"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    label: str
    params_b: float
    quant: str | None = None
    cpu_embeddings: bool = False
    set_attention: bool = False


MODELS = [
    ModelSpec("qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B", "Qwen3.5 0.8B", 0.87),
    ModelSpec("lfm2.5-1.2b", "LiquidAI/LFM2.5-1.2B-Instruct", "LFM2.5 1.2B Instruct", 1.17),
    ModelSpec("qwen3.5-2b", "Qwen/Qwen3.5-2B", "Qwen3.5 2B", 2.27),
    ModelSpec("granite-4.2-3b", "ibm-granite/granite-4.2-3b", "Granite 4.2 3B (4-bit)", 3.66, "nf4",
              set_attention=True),
    ModelSpec("qwen3.5-4b", "Qwen/Qwen3.5-4B", "Qwen3.5 4B (4-bit)", 4.66, "nf4"),
    ModelSpec("qwen3.5-9b", "Qwen/Qwen3.5-9B", "Qwen3.5 9B (4-bit)", 9.65, "nf4", cpu_embeddings=True),
]
MODELS_BY_KEY = {m.key: m for m in MODELS}


@dataclass
class Question:
    """choice: options (+ "none" if none_option); score: ordered levels, low to high; yesno: Sì/No."""
    question: str
    options: list[str] = field(default_factory=list)
    answer: str | None = None
    type: str = "choice"
    none_option: bool = True
    state: str | None = None
    id: str | None = None

    @property
    def all_options(self) -> list[str]:
        if self.type == "yesno":
            return list(YES_NO)
        if self.type == "choice" and self.none_option:
            return self.options + [NONE_OPTION]
        return list(self.options)

    def validate(self):
        if self.type not in QUESTION_TYPES:
            raise ValueError(f"Tipo sconosciuto: {self.type}")
        n = len(self.all_options)
        if n < 2 or n > len(LETTERS):
            raise ValueError(f"Servono da 2 a {len(LETTERS)} opzioni: {self.question!r}")
        if self.type == "score" and n > MAX_SCORE_LEVELS:
            raise ValueError(f"Massimo {MAX_SCORE_LEVELS} livelli: {self.question!r}")
        if self.answer and (len(self.answer) != 1 or self.answer not in LETTERS[:n]):
            raise ValueError(f"Risposta {self.answer!r} non valida per {self.question!r}")


@dataclass
class Calibration:
    """Split conformal prediction (KnowNo, Ren et al. 2023): score = 1 - p(correct option)."""
    epsilon: float
    qhat: float
    n: int
    key: str
    created: float = field(default_factory=time.time)

    @property
    def threshold(self) -> float:
        return 1 - self.qhat


def calibrate(probs_true: list[float], epsilon: float, key: str) -> Calibration:
    n = len(probs_true)
    if n == 0:
        raise ValueError("Servono domande con risposta nota per calibrare")
    scores = sorted(1 - p for p in probs_true)
    rank = math.ceil((n + 1) * (1 - epsilon))
    qhat = 1.0 if rank > n else scores[rank - 1]
    return Calibration(epsilon=epsilon, qhat=qhat, n=n, key=key)


def calibration_key(model: str, prompt: str, scramble: int, aggregation: str, set_attention: bool = False) -> str:
    key = f"{model}|{prompt}|s{scramble}|{aggregation if scramble > 1 else 'single'}"
    return key + "|set" if set_attention else key


def permutations_for(n: int, k: int, seed: int) -> list[list[int]]:
    """Cyclic shifts first (each option visits each position once), then random permutations."""
    base = list(range(n))
    perms = [base[s:] + base[:s] for s in range(min(k, n))]
    rng = random.Random(seed)
    seen = {tuple(p) for p in perms}
    while len(perms) < k and len(seen) < math.factorial(n):
        p = base[:]
        rng.shuffle(p)
        if tuple(p) not in seen:
            seen.add(tuple(p))
            perms.append(p)
    return perms


def aggregate(variant_probs: torch.Tensor, method: str, weights: torch.Tensor | None = None) -> torch.Tensor:
    """variant_probs: [k, n_options] in original option order; weights: [k], used by "weighted"."""
    if method == "mean":
        return variant_probs.mean(0)
    if method == "weighted":
        w = torch.ones(len(variant_probs)) if weights is None else weights.clamp_min(1e-6)
        return (w[:, None] * variant_probs).sum(0) / w.sum()
    if method == "geomean":
        return variant_probs.clamp_min(1e-12).log().mean(0).softmax(-1)
    if method == "vote":
        votes = torch.bincount(variant_probs.argmax(-1), minlength=variant_probs.shape[1]).float()
        scores = votes + 1e-3 * variant_probs.mean(0)
        return scores / scores.sum()
    raise ValueError(f"Aggregazione sconosciuta: {method}")


def build_report(
    questions: list[Question],
    variants: list[list[dict]],
    aggregation: str,
    calibration: Calibration | None = None,
) -> dict:
    """Aggregates per-variant probabilities into per-question results and a summary."""
    results = []
    for q, vs in zip(questions, variants):
        vp = torch.tensor([v["probs"] for v in vs])
        mass = torch.tensor([v["letter_mass"] for v in vs])
        final = aggregate(vp, aggregation, mass) if len(vs) > 1 else vp[0]
        pred = int(final.argmax())
        item = {
            "id": q.id,
            "type": q.type,
            "question": q.question,
            "options": q.all_options,
            "probs": final.tolist(),
            "argmax": LETTERS[pred],
            "confidence": float(final[pred]),
            "entropy": float(-(final * final.clamp_min(1e-12).log()).sum()),
            "consistency": float((vp.argmax(-1) == pred).float().mean()),
            "std": vp.std(0, unbiased=False).tolist() if len(vs) > 1 else None,
            "letter_mass": sum(v["letter_mass"] for v in vs) / len(vs),
            "answer": q.answer,
            "correct": q.answer == LETTERS[pred] if q.answer else None,
            "variants": vs,
        }
        if q.type == "score":
            levels = torch.arange(1, len(final) + 1, dtype=final.dtype)
            item["score"] = float((final * levels).sum())
            item["score_norm"] = (item["score"] - 1) / (len(final) - 1)
        elif q.type == "yesno":
            item["p_yes"] = float(final[0])
        if calibration:
            pset = [LETTERS[i] for i, p in enumerate(final.tolist()) if p >= calibration.threshold]
            item["prediction_set"] = pset or [LETTERS[pred]]
            item["decided"] = len(item["prediction_set"]) == 1
        results.append(item)

    graded = [r for r in results if r["answer"]]
    summary = {
        "questions": len(results),
        "graded": len(graded),
        "accuracy": sum(r["correct"] for r in graded) / len(graded) if graded else None,
        **calibration_metrics(graded),
        "avg_letter_mass": sum(r["letter_mass"] for r in results) / len(results),
        "avg_confidence": sum(r["confidence"] for r in results) / len(results),
        "avg_consistency": sum(r["consistency"] for r in results) / len(results),
    }
    if calibration:
        decided = [r for r in results if r["decided"]]
        summary["conformal"] = {
            "epsilon": calibration.epsilon,
            "qhat": calibration.qhat,
            "threshold": calibration.threshold,
            "calibration_n": calibration.n,
            "decided_rate": len(decided) / len(results),
            "avg_set_size": sum(len(r["prediction_set"]) for r in results) / len(results),
            "coverage": (sum(r["answer"] in r["prediction_set"] for r in graded) / len(graded)
                         if graded else None),
            "accuracy_when_decided": (
                sum(r["correct"] for r in decided if r["answer"]) / sum(1 for r in decided if r["answer"])
                if any(r["answer"] for r in decided) else None),
        }
    return {"summary": summary, "results": results}


def calibration_metrics(graded: list[dict], bins: int = 10) -> dict:
    """ECE on the top-1 confidence, multi-class Brier score and NLL of the correct option."""
    if not graded:
        return {"ece": None, "brier": None, "nll": None, "reliability": []}
    ece, reliability = 0.0, []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        rows = [r for r in graded if lo < r["confidence"] <= hi or (b == 0 and r["confidence"] == 0)]
        if not rows:
            continue
        conf = sum(r["confidence"] for r in rows) / len(rows)
        acc = sum(r["correct"] for r in rows) / len(rows)
        ece += len(rows) / len(graded) * abs(acc - conf)
        reliability.append({"lo": lo, "hi": hi, "confidence": conf, "accuracy": acc, "n": len(rows)})
    brier, nll = 0.0, 0.0
    for r in graded:
        t = LETTERS.index(r["answer"])
        brier += sum((p - (i == t)) ** 2 for i, p in enumerate(r["probs"]))
        nll -= math.log(max(r["probs"][t], 1e-12))
    return {"ece": ece, "brier": brier / len(graded), "nll": nll / len(graded), "reliability": reliability}


def true_probs(report: dict) -> list[float]:
    return [r["probs"][LETTERS.index(r["answer"])] for r in report["results"] if r["answer"]]


def _embeddings_to_cpu(model):
    """Replaces accelerate's offload hook (which streams the table to the GPU) with a plain CPU lookup."""
    from accelerate.hooks import remove_hook_from_module
    emb = model.get_input_embeddings()
    weight = emb._hf_hook.weights_map["weight"]
    remove_hook_from_module(emb)
    model.set_input_embeddings(torch.nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=emb.padding_idx))


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()



BUCKET = 32
DEFAULT_BATCH_TOKENS = 2048
MIN_PREFIX_SAVING = 0.25  # below this, copying the cache costs more than the skipped prefill (bench stress)
DEFAULT_BATCH_ROWS = 64
_OPTIONS_MARK = "<<<DECISIONLLM_OPTIONS>>>"
_PAD_SEG, _PREFIX_SEG, _SUFFIX_SEG = -2, -1, 1_000_000


def _bucket(n: int) -> int:
    """Rounds lengths up so tensor shapes repeat and Triton kernels are not re-tuned for every batch."""
    return -(-n // BUCKET) * BUCKET


def _lcp(seqs: list[list[int]]) -> int:
    n = min(len(s) for s in seqs)
    for i in range(n):
        t = seqs[0][i]
        if any(s[i] != t for s in seqs[1:]):
            return i
    return n


def _shared_prefix(seqs: list[list[int]]) -> int:
    """Common prefix length, leaving at least one token per sequence to read the logits from."""
    return min(_lcp(seqs), min(len(s) for s in seqs) - 1)


def _token_batches(cost, items: list, max_rows: int, max_tokens: int) -> list[list]:
    """Greedy chunks of `items` (already sorted) such that cost(chunk) = (rows, padded tokens) fits the budgets."""
    chunks, chunk = [], []
    for item in items:
        rows, tokens = cost(chunk + [item])
        if chunk and (rows > max_rows or tokens > max_tokens):
            chunks.append(chunk)
            chunk = []
        chunk.append(item)
    if chunk:
        chunks.append(chunk)
    return chunks


def _select_cache(cache, idx: torch.Tensor):
    """Gathers (and repeats) batch rows of a transformers cache, for attention and linear-attention layers."""
    for layer in cache.layers:
        keys = getattr(layer, "keys", None)
        if isinstance(keys, torch.Tensor) and keys.numel():
            layer.keys = keys.index_select(0, idx.to(keys.device))
            layer.values = layer.values.index_select(0, idx.to(keys.device))
        for states in (getattr(layer, "conv_states", None), getattr(layer, "recurrent_states", None)):
            for k, t in (states or {}).items():
                if isinstance(t, torch.Tensor):
                    states[k] = t.index_select(0, idx.to(t.device))


class Engine:
    def __init__(self):
        self.spec: ModelSpec | None = None
        self.model = None
        self.tokenizer = None
        self.load_seconds = 0.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._letter_ids: dict[str, list[int]] = {}

    def unload(self):
        self.model = None
        self.tokenizer = None
        self.spec = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, key: str) -> dict:
        if self.spec and self.spec.key == key:
            return self.info()
        spec = MODELS_BY_KEY[key]
        self.unload()
        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(spec.repo, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        kwargs = {"dtype": dtype, "device_map": self.device}
        if spec.cpu_embeddings and self.device == "cuda":
            # embedding lookup is cheap on CPU and frees ~2 GB of VRAM for large-vocabulary models
            kwargs["device_map"] = {"model.embed_tokens": "cpu", "model.layers": 0, "model.norm": 0,
                                    "model.rotary_emb": 0, "lm_head": 0}
        if spec.quant == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=spec.cpu_embeddings,
            )
        model = AutoModelForCausalLM.from_pretrained(spec.repo, **kwargs).eval()
        if spec.cpu_embeddings and self.device == "cuda":
            _embeddings_to_cpu(model)
        _sync()
        self.load_seconds = time.perf_counter() - t0
        self.spec, self.model, self.tokenizer = spec, model, tok
        self._letter_ids = {}
        return self.info()

    def _model_inputs(self, ids: torch.Tensor) -> dict:
        if self.spec.cpu_embeddings and self.device == "cuda":
            return {"inputs_embeds": self.model.get_input_embeddings()(ids).to(self.device)}
        return {"input_ids": ids.to(self.device)}

    def info(self) -> dict:
        if not self.spec:
            return {"loaded": None}
        mem = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        return {
            "loaded": self.spec.key,
            "label": self.spec.label,
            "repo": self.spec.repo,
            "load_seconds": round(self.load_seconds, 2),
            "vram_gb": round(mem, 2),
            "device": self.device,
        }

    def letter_ids(self, letter: str) -> list[int]:
        """Token ids that spell the answer letter, with and without a leading space."""
        if letter not in self._letter_ids:
            ids = set()
            for variant in (letter, " " + letter):
                toks = self.tokenizer.encode(variant, add_special_tokens=False)
                if len(toks) == 1:
                    ids.add(toks[0])
            self._letter_ids[letter] = sorted(ids)
        return self._letter_ids[letter]

    def _prompt_parts(self, q: Question, options: list[str], style: str) -> tuple[str, list[str], str]:
        """(prefix, one segment per option, suffix): their concatenation is the full prompt."""
        ps = PROMPT_STYLES_BY_KEY[style]
        context = f"{ps.context_label}:\n{q.state.strip()}\n\n" if q.state and q.state.strip() else ""
        msgs = [{"role": "user", "content": f"{context}{q.question}\n\n{_OPTIONS_MARK}\n\n{ps.render(len(options))}"}]
        rendered = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prefix, suffix = rendered.split(_OPTIONS_MARK)
        segments = [f"{LETTERS[i]}) {o}" + ("\n" if i < len(options) - 1 else "") for i, o in enumerate(options)]
        return prefix, segments, suffix + ps.prefix

    def build_prompt(self, q: Question, options: list[str], style: str = DEFAULT_PROMPT) -> str:
        prefix, segments, suffix = self._prompt_parts(q, options, style)
        return prefix + "".join(segments) + suffix

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _letter_table(self) -> list[list[int]]:
        return [self.letter_ids(letter) for letter in LETTERS]

    def _store(self, rows: list[int], logits: torch.Tensor, letters: torch.Tensor, tops: torch.Tensor):
        """Keeps only the letter log-probs and the unconstrained argmax of each row."""
        logp = logits.float().log_softmax(-1)
        letters[rows] = torch.stack([logp[:, ids].logsumexp(-1) for ids in self._letter_table()], -1).cpu()
        tops[rows] = logits.argmax(-1).cpu()

    def _head(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.model.get_output_embeddings()(hidden)
        scaling = getattr(self.model.config, "logits_scaling", None)
        return logits / scaling if scaling else logits

    def _run_chunks(self, chunks: list[list], run, stats: dict):
        """Runs each chunk; on OOM the chunk is split in two and retried."""
        queue = list(chunks)
        while queue:
            chunk = queue.pop(0)
            try:
                _sync()
                t = time.perf_counter()
                with torch.inference_mode():
                    run(chunk)
                _sync()
                stats["forward_s"] += time.perf_counter() - t
                stats["batches"] += 1
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(chunk) == 1:
                    raise
                stats["oom_retries"] += 1
                queue[:0] = [chunk[: len(chunk) // 2], chunk[len(chunk) // 2 :]]

    def _forward_plain(self, enc, max_rows, max_tokens, stats, letters, tops):
        lengths = [len(e) for e in enc]
        pad = self.tokenizer.pad_token_id

        def run(chunk):
            width = _bucket(max(lengths[j] for j in chunk))
            ids = torch.full((len(chunk), width), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for r, j in enumerate(chunk):
                ids[r, width - lengths[j]:] = torch.tensor(enc[j])
                mask[r, width - lengths[j]:] = 1
            out = self.model(**self._model_inputs(ids), attention_mask=mask.to(self.device), logits_to_keep=1,
                             use_cache=False)
            self._store(chunk, out.logits[:, -1], letters, tops)
            stats["tokens"] += int(mask.sum())
            stats["padded_tokens"] += ids.numel()
            stats["max_rows"] = max(stats["max_rows"], len(chunk))

        order = sorted(range(len(enc)), key=lengths.__getitem__)
        cost = lambda c: (len(c), len(c) * _bucket(max(lengths[j] for j in c)))  # noqa: E731
        self._run_chunks(_token_batches(cost, order, max_rows, max_tokens), run, stats)

    def _forward_set(self, parts, max_rows, max_tokens, stats, letters, tops):
        """Set-based prompting: option segments share positions and cannot see each other, so their order
        does not matter (McIlroy-Young et al. 2024). Needs attention-only models."""
        rows = []
        for prefix, segments, suffix in parts:
            ids, pos, seg = list(prefix), list(range(len(prefix))), [_PREFIX_SEG] * len(prefix)
            start = len(prefix)
            for k, s in enumerate(segments):
                ids += s
                pos += range(start, start + len(s))
                seg += [k] * len(s)
            end = start + max(len(s) for s in segments)
            ids += suffix
            pos += range(end, end + len(suffix))
            seg += [_SUFFIX_SEG] * len(suffix)
            rows.append((ids, pos, seg))
        lengths = [len(r[0]) for r in rows]
        pad = self.tokenizer.pad_token_id

        def run(chunk):
            width = _bucket(max(lengths[j] for j in chunk))
            ids = torch.full((len(chunk), width), pad, dtype=torch.long)
            pos = torch.zeros_like(ids)
            seg = torch.full_like(ids, _PAD_SEG)
            for r, j in enumerate(chunk):
                off = width - lengths[j]
                ids[r, off:], pos[r, off:], seg[r, off:] = (torch.tensor(x) for x in rows[j])
            causal = torch.ones(width, width, dtype=torch.bool).tril()
            visible = (seg[:, :, None] == seg[:, None, :]) | (seg == _PREFIX_SEG)[:, None, :] | (seg == _SUFFIX_SEG)[:, :, None]
            allowed = (causal & visible & (seg != _PAD_SEG)[:, None, :]) | torch.eye(width, dtype=torch.bool)
            out = self.model(**self._model_inputs(ids), attention_mask=allowed[:, None].to(self.device),
                             position_ids=pos.to(self.device), logits_to_keep=1, use_cache=False)
            self._store(chunk, out.logits[:, -1], letters, tops)
            stats["tokens"] += sum(lengths[j] for j in chunk)
            stats["padded_tokens"] += ids.numel()
            stats["max_rows"] = max(stats["max_rows"], len(chunk))

        order = sorted(range(len(rows)), key=lengths.__getitem__)
        cost = lambda c: (len(c), len(c) * _bucket(max(lengths[j] for j in c)))  # noqa: E731
        self._run_chunks(_token_batches(cost, order, max_rows, max_tokens), run, stats)

    def _forward_prefix(self, enc, groups, max_rows, max_tokens, stats, letters, tops) -> bool:
        """Prefix caching: each shared prefix is computed once, then the cache is repeated for the suffixes.
        Uses one prefix per group (the permutations of a question) or one prefix for everything (a shared
        state), whichever skips more tokens. Returns False when there is nothing worth sharing."""
        everything = [j for g in groups for j in g]
        per_group = [_shared_prefix([enc[j] for j in g]) for g in groups]
        saved_groups = sum((len(g) - 1) * p for g, p in zip(groups, per_group))
        shared = _shared_prefix([enc[j] for j in everything])
        saved_shared = (len(everything) - 1) * shared
        total = sum(len(e) for e in enc)
        if max(saved_groups, saved_shared) < MIN_PREFIX_SAVING * total:
            return False
        if saved_shared > saved_groups:
            groups, per_group = [everything], [shared]
            stats["prefix_mode"] = "shared"
        else:
            stats["prefix_mode"] = "per-question"
        stats["tokens_saved"] = max(saved_groups, saved_shared)
        pad = self.tokenizer.pad_token_id
        base = self.model.model

        # one row per item, rows of a group adjacent: a batch computes each prefix once and OOM can split any batch
        items = [(gi, [j]) for gi in sorted(range(len(groups)), key=lambda g: (per_group[g], len(enc[groups[g][0]])))
                 for j in groups[gi]]

        def suffix_len(j, gi):
            return len(enc[j]) - per_group[gi]

        def cost(chunk):
            rows = sum(len(r) for _, r in chunk)
            prefix = len({gi for gi, _ in chunk}) * _bucket(max(per_group[gi] for gi, _ in chunk))
            return rows, prefix + rows * _bucket(max(suffix_len(j, gi) for gi, r in chunk for j in r))

        def prefix_cache(gis):
            width = _bucket(max(per_group[gi] for gi in gis))
            ids = torch.full((len(gis), width), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for r, gi in enumerate(gis):
                n = per_group[gi]
                ids[r, width - n:] = torch.tensor(enc[groups[gi][0]][:n])
                mask[r, width - n:] = 1
            out = base(**self._model_inputs(ids), attention_mask=mask.to(self.device), use_cache=True)
            stats["tokens"] += int(mask.sum())
            stats["padded_tokens"] += ids.numel()
            return out.past_key_values, mask

        shared_cache = None
        if stats["prefix_mode"] == "shared":
            _sync()
            t = time.perf_counter()
            with torch.inference_mode():
                shared_cache = prefix_cache([0])
            _sync()
            stats["forward_s"] += time.perf_counter() - t

        def run(chunk):
            gis = list(dict.fromkeys(gi for gi, _ in chunk))
            if shared_cache:
                cache, prefix_mask = copy.deepcopy(shared_cache[0]), shared_cache[1]
            else:
                cache, prefix_mask = prefix_cache(gis)
            rows = [(gis.index(gi), gi, j) for gi, r in chunk for j in r]
            idx = torch.tensor([g for g, _, _ in rows])
            _select_cache(cache, idx)
            width = _bucket(max(suffix_len(j, gi) for _, gi, j in rows))
            ids = torch.full((len(rows), width), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for r, (_, gi, j) in enumerate(rows):
                s = enc[j][per_group[gi]:]
                ids[r, : len(s)] = torch.tensor(s)
                mask[r, : len(s)] = 1
            full_mask = torch.cat([prefix_mask.index_select(0, idx), mask], 1)
            out = base(**self._model_inputs(ids), attention_mask=full_mask.to(self.device), past_key_values=cache,
                       use_cache=True)
            last = torch.tensor([suffix_len(j, gi) - 1 for _, gi, j in rows], device=out.last_hidden_state.device)
            hidden = out.last_hidden_state[torch.arange(len(rows), device=last.device), last]
            self._store([j for _, _, j in rows], self._head(hidden), letters, tops)
            stats["tokens"] += int(mask.sum())
            stats["padded_tokens"] += ids.numel()
            stats["max_rows"] = max(stats["max_rows"], len(rows))

        self._run_chunks(_token_batches(cost, items, max_rows, max_tokens), run, stats)
        return True

    def run_variants(
        self,
        questions: list[Question],
        scramble: int = 1,
        batch_size: int = DEFAULT_BATCH_ROWS,
        prompt_style: str = DEFAULT_PROMPT,
        seed: int = 0,
        max_questions: int | None = MAX_QUESTIONS,
        prefix_cache: bool = False,
        set_attention: bool = False,
        batch_tokens: int = DEFAULT_BATCH_TOKENS,
    ) -> tuple[list[list[dict]], dict]:
        """One forward pass per (question, permutation). With scramble > 1 the
        "none" option is shuffled too, as in KnowNo."""
        if self.model is None:
            raise RuntimeError("Nessun modello caricato")
        if not questions:
            raise ValueError("Nessuna domanda")
        if max_questions and len(questions) > max_questions:
            raise ValueError(f"Massimo {max_questions} domande per richiesta")
        if set_attention and not self.spec.set_attention:
            raise ValueError(f"{self.spec.label} non supporta il set-based attention (serve un modello solo attention)")
        for q in questions:
            q.validate()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()

        parts, meta, groups = [], [], []
        for qi, q in enumerate(questions):
            n = len(q.all_options)
            perms = [list(range(n))] if scramble <= 1 else permutations_for(n, scramble, seed + qi)
            groups.append(list(range(len(meta), len(meta) + len(perms))))
            for order in perms:
                parts.append(self._prompt_parts(q, [q.all_options[o] for o in order], prompt_style))
                meta.append((qi, order))

        stats = {"tokenize_s": 0.0, "forward_s": 0.0, "batches": 0, "tokens": 0, "padded_tokens": 0,
                 "oom_retries": 0, "max_rows": 0, "prefix_mode": "off", "tokens_saved": 0}
        t0 = time.perf_counter()
        if set_attention:
            enc = [(self._encode(p), [self._encode(s) for s in segs], self._encode(x)) for p, segs, x in parts]
        else:
            enc = [self._encode(p + "".join(segs) + x) for p, segs, x in parts]
        stats["tokenize_s"] = time.perf_counter() - t0

        letters = torch.empty(len(parts), len(LETTERS))
        tops = torch.empty(len(parts), dtype=torch.long)
        if set_attention:
            stats["prefix_mode"] = "set-attention"
            self._forward_set(enc, batch_size, batch_tokens, stats, letters, tops)
        elif not (prefix_cache and self._forward_prefix(enc, groups, batch_size, batch_tokens, stats, letters, tops)):
            self._forward_plain(enc, batch_size, batch_tokens, stats, letters, tops)

        max_n = max(len(q.all_options) for q in questions)
        variants: list[list[dict]] = [[] for _ in questions]
        position_sum = torch.zeros(max_n)
        position_cnt = torch.zeros(max_n)
        for row, top, (qi, order) in zip(letters, tops, meta):
            n = len(order)
            letter_logp = row[:n]
            pos_probs = letter_logp.softmax(-1)
            position_sum[:n] += pos_probs
            position_cnt[:n] += 1
            orig = torch.empty(n)
            orig[torch.tensor(order)] = pos_probs
            variants[qi].append({
                "order": order,
                "position_probs": pos_probs.tolist(),
                "probs": orig.tolist(),
                "letter_mass": float(letter_logp.exp().sum()),
                "unconstrained_top": self.tokenizer.decode([int(top)]),
            })

        forward = stats["forward_s"]
        timing = {
            "prompts": len(parts),
            "total_s": time.perf_counter() - t_start,
            "tokenize_s": stats["tokenize_s"],
            "forward_s": forward,
            "prompts_per_s": len(parts) / forward if forward else None,
            "tokens": stats["tokens"],
            "tokens_per_s": stats["tokens"] / forward if forward else None,
            "padding_ratio": 1 - stats["tokens"] / stats["padded_tokens"],
            "batches": stats["batches"],
            "max_batch_rows": stats["max_rows"],
            "oom_retries": stats["oom_retries"],
            "prefix_mode": stats["prefix_mode"],
            "tokens_saved": stats["tokens_saved"],
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
            "position_bias": (position_sum / position_cnt.clamp_min(1)).tolist(),
            "sample_prompt": "".join([parts[0][0], *parts[0][1], parts[0][2]]),
        }
        return variants, timing

    def score(
        self,
        questions: list[Question],
        scramble: int = 1,
        aggregation: str = "geomean",
        batch_size: int = DEFAULT_BATCH_ROWS,
        prompt_style: str = DEFAULT_PROMPT,
        calibration: Calibration | None = None,
        seed: int = 0,
        prefix_cache: bool = False,
        set_attention: bool = False,
        batch_tokens: int = DEFAULT_BATCH_TOKENS,
    ) -> dict:
        variants, timing = self.run_variants(questions, scramble, batch_size, prompt_style, seed,
                                             prefix_cache=prefix_cache, set_attention=set_attention,
                                             batch_tokens=batch_tokens)
        report = build_report(questions, variants, aggregation, calibration)
        report["summary"].update({k: v for k, v in timing.items() if k != "sample_prompt"})
        report["sample_prompt"] = timing["sample_prompt"]
        report["model"] = self.info()
        report["params"] = {"scramble": scramble, "aggregation": aggregation, "batch_size": batch_size,
                            "batch_tokens": batch_tokens, "prompt_style": prompt_style, "seed": seed,
                            "prefix_cache": prefix_cache, "set_attention": set_attention}
        return report
