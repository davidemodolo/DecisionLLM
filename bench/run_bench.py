"""Benchmark of all models on Global-MMLU-Lite (Italian) and on the local test set.

Phases:
  prompt  compare prompt styles on the dev split (no scramble)
  eval    dev -> conformal calibration, test -> accuracy / coverage, with and without scramble
  stress  128 local questions x scramble 5 at several batch sizes (timing and peak VRAM)
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import (  # noqa: E402
    AGGREGATIONS, DEFAULT_PROMPT, LETTERS, MODELS, MODELS_BY_KEY, PROMPT_STYLES, Engine, Question,
    build_report, calibrate, true_probs,
)

DATA = ROOT / "bench" / "data"
RESULTS = ROOT / "bench" / "results"
NONE_RATE = 0.2
EPSILON = 0.1
SCRAMBLE = 5
BATCH_SIZE = {"qwen3.5-9b": 16}
STRESS_BUDGETS = {"qwen3.5-9b": [1024, 2048, 4096]}
DEFAULT_STRESS_BUDGETS = [2048, 4096, 8192]


def load_split(split: str) -> list[dict]:
    """Global-MMLU-Lite rows; NONE_RATE of them lose the correct option, so "none" becomes correct."""
    df = pd.read_parquet(DATA / f"global_mmlu_lite_it_{split}.parquet")
    rng = random.Random(f"nota-{split}")
    items = []
    for row in df.itertuples():
        options = [row.option_a, row.option_b, row.option_c, row.option_d]
        answer = row.answer
        nota = rng.random() < NONE_RATE
        if nota:
            options.pop("ABCD".index(answer))
            answer = LETTERS[len(options)]
        items.append({"question": Question(row.question, options, answer), "nota": nota,
                      "category": row.subject_category})
    return items


def local_questions() -> list[Question]:
    raw = json.loads((ROOT / "data" / "test_questions.json").read_text())
    return [Question(q["question"], q["options"], q["answer"]) for q in raw]


def accuracy(results: list[dict], mask: list[bool] | None = None) -> float | None:
    rows = [r for r, m in zip(results, mask or [True] * len(results)) if m]
    return sum(r["correct"] for r in rows) / len(rows) if rows else None


def save(name: str, data: dict):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))


def phase_prompt(engine: Engine, model: str, batch_size: int) -> dict:
    items = load_split("dev")
    qs = [i["question"] for i in items]
    out = {}
    for ps in PROMPT_STYLES:
        variants, timing = engine.run_variants(qs, 1, batch_size, ps.key, max_questions=None)
        rep = build_report(qs, variants, "mean")
        res = rep["results"]
        out[ps.key] = {
            "accuracy": rep["summary"]["accuracy"],
            "accuracy_std": accuracy(res, [not i["nota"] for i in items]),
            "accuracy_nota": accuracy(res, [i["nota"] for i in items]),
            "letter_mass": rep["summary"]["avg_letter_mass"],
            "position_bias": timing["position_bias"],
            "forward_s": timing["forward_s"],
        }
        print(f"  {ps.key:12s} acc={out[ps.key]['accuracy']:.3f} mass={out[ps.key]['letter_mass']:.3f}", flush=True)
    return out


def configs(variants: list[str]):
    for v in variants:
        yield 1, "single", v
        for agg in AGGREGATIONS:
            yield SCRAMBLE, agg, v


def unscrambled(variants: list[list[dict]]) -> list[list[dict]]:
    """The first cyclic permutation is the original order, so the unscrambled run is variant 0."""
    return [vs[:1] for vs in variants]


def position_bias(variants: list[list[dict]]) -> list[float]:
    rows = [v["position_probs"] for vs in variants for v in vs]
    n = max(len(r) for r in rows)
    return [sum(r[i] for r in rows if len(r) > i) / sum(1 for r in rows if len(r) > i) for i in range(n)]


def phase_eval(engine: Engine, model: str, batch_size: int, prompt: str) -> dict:
    dev, test = load_split("dev"), load_split("test")
    dev_q, test_q = [i["question"] for i in dev], [i["question"] for i in test]
    nota_mask = [i["nota"] for i in test]
    modes = ["plain"] + (["set"] if engine.spec.set_attention else [])
    runs, raw = {}, {}
    for mode in modes:
        kw = {"max_questions": None, "prefix_cache": mode == "plain", "set_attention": mode == "set"}
        dv, _ = engine.run_variants(dev_q, SCRAMBLE, batch_size, prompt, **kw)
        tv, timing = engine.run_variants(test_q, SCRAMBLE, batch_size, prompt, **kw)
        runs[(SCRAMBLE, mode)] = (dv, tv, timing)
        runs[(1, mode)] = (unscrambled(dv), unscrambled(tv), timing)
    for (scramble, mode), (dev_v, test_v, _) in runs.items():
        raw[f"s{scramble}" + ("-set" if mode == "set" else "")] = {
            split: [{"answer": q.answer, "nota": it["nota"], "category": it["category"],
                     "probs": [v["probs"] for v in vs], "letter_mass": [v["letter_mass"] for v in vs]}
                    for q, it, vs in zip(qs, items, variants)]
            for split, qs, items, variants in (("dev", dev_q, dev, dev_v), ("test", test_q, test, test_v))
        }
    save(f"raw-{model}", raw)

    out = {"prompt": prompt, "epsilon": EPSILON, "configs": {}}
    for scramble, agg, mode in configs(modes):
        dv, tv, timing = runs[(scramble, mode)]
        method = "mean" if agg == "single" else agg
        name = f"s{scramble}-{agg}" + ("-set" if mode == "set" else "")
        dev_rep = build_report(dev_q, dv, method)
        cal = calibrate(true_probs(dev_rep), EPSILON, f"{model}|{prompt}|{name}")
        rep = build_report(test_q, tv, method, cal)
        s, res = rep["summary"], rep["results"]
        out["configs"][name] = {
            "accuracy": s["accuracy"],
            "accuracy_std": accuracy(res, [not m for m in nota_mask]),
            "accuracy_nota": accuracy(res, nota_mask),
            "letter_mass": s["avg_letter_mass"],
            "avg_confidence": s["avg_confidence"],
            "avg_consistency": s["avg_consistency"],
            "ece": s["ece"],
            "brier": s["brier"],
            "nll": s["nll"],
            "reliability": s["reliability"],
            "conformal": s["conformal"],
            "position_bias": position_bias(tv),
            "prompts": sum(len(vs) for vs in tv),
        }
        c = s["conformal"]
        print(f"  {name:18s} acc={s['accuracy']:.3f} (std {out['configs'][name]['accuracy_std']:.3f}, "
              f"nota {out['configs'][name]['accuracy_nota']:.3f}) cov={c['coverage']:.3f} "
              f"decided={c['decided_rate']:.2f} acc@decided={c['accuracy_when_decided'] or 0:.3f} "
              f"set={c['avg_set_size']:.2f} ece={s['ece']:.3f} brier={s['brier']:.3f}", flush=True)
    return out


STATE_TEXT = (
    "Segnalazione ricevuta il 3 marzo. Il cliente scrive: ho acquistato un frigorifero combinato a gennaio presso il "
    "vostro punto vendita di via Roma. Dopo tre settimane il congelatore ha smesso di raffreddare e il cibo si è "
    "scongelato due volte. Ho chiamato l'assistenza il 12 febbraio: mi hanno promesso un tecnico entro cinque giorni "
    "lavorativi, ma non si è presentato nessuno. Ho richiamato il 20 febbraio e il 27 febbraio, ogni volta con un "
    "operatore diverso che non trovava la pratica precedente. Il frigorifero è ancora in garanzia e ho conservato lo "
    "scontrino e il certificato di garanzia. Ho dovuto comprare un piccolo congelatore a pozzetto per non buttare la "
    "spesa, spendendo 180 euro. Chiedo la sostituzione dell'apparecchio oppure il rimborso completo, oltre al "
    "rimborso della spesa per il congelatore temporaneo. Se non riceverò risposta entro dieci giorni mi rivolgerò a "
    "un'associazione di consumatori. Allego le foto del display che mostra l'errore E4 e le ricevute delle chiamate."
)
STATE_QUESTIONS = [
    ("Qual è il problema principale?", "choice", ["Consegna", "Guasto del prodotto", "Fatturazione", "Cortesia"]),
    ("Quale reparto deve gestire la pratica?", "choice", ["Vendite", "Assistenza tecnica", "Amministrazione", "Logistica"]),
    ("Quanto è urgente la pratica?", "score", ["Bassa", "Media", "Alta", "Critica"]),
    ("Quanto è arrabbiato il cliente?", "score", ["Per niente", "Poco", "Abbastanza", "Molto"]),
    ("Il prodotto è in garanzia?", "yesno", []),
    ("Il cliente chiede un rimborso?", "yesno", []),
    ("Il cliente minaccia azioni esterne?", "yesno", []),
    ("Il cliente ha allegato documenti?", "yesno", []),
    ("Il tecnico si è presentato?", "yesno", []),
    ("Che tipo di prodotto è coinvolto?", "choice", ["Lavatrice", "Frigorifero", "Forno", "Televisore"]),
    ("Quante volte il cliente ha richiamato?", "choice", ["Nessuna", "Una", "Due", "Cinque"]),
    ("Qual è la richiesta principale?", "choice", ["Informazioni", "Sostituzione o rimborso", "Sconto", "Recesso"]),
    ("Il cliente ha sostenuto spese aggiuntive?", "yesno", []),
    ("Quanto è dettagliata la segnalazione?", "score", ["Vaga", "Sufficiente", "Dettagliata", "Molto dettagliata"]),
    ("Il cliente cita un codice di errore?", "yesno", []),
    ("Entro quanto il cliente si aspetta risposta?", "choice", ["Un giorno", "Cinque giorni", "Dieci giorni", "Un mese"]),
    ("La pratica precedente era tracciata correttamente?", "yesno", []),
    ("Qual è il canale di acquisto?", "choice", ["Online", "Negozio fisico", "Telefono", "Rivenditore terzo"]),
    ("Rischio di perdere il cliente?", "score", ["Basso", "Medio", "Alto"]),
    ("Serve un intervento a domicilio?", "yesno", []),
]


def state_questions() -> list[Question]:
    return [Question(q, opts, type=t, none_option=False, state=STATE_TEXT) for q, t, opts in STATE_QUESTIONS]


def hot(engine: Engine, qs, **kw) -> dict:
    """Runs twice and returns the second timing: Triton kernels are compiled and tuned on the first run."""
    engine.run_variants(qs, **kw)
    torch.cuda.empty_cache()
    _, timing = engine.run_variants(qs, **kw)
    return {k: timing[k] for k in ("prompts", "forward_s", "prompts_per_s", "tokens", "tokens_per_s", "peak_vram_gb",
                                   "max_batch_rows", "oom_retries", "padding_ratio", "prefix_mode", "tokens_saved")}


def phase_stress(engine: Engine, model: str, prompt: str, budgets: list[int]) -> dict:
    """Hot timings for 128 questions (every other one) x 5 permutations and for one long state with 20 questions x 5 permutations."""
    out = {"budgets": {}, "prefix_cache": {}, "state": {}}
    qs = local_questions()[::2]
    base = {"scramble": SCRAMBLE, "prompt_style": prompt, "batch_size": 1024}

    def report(label, t):
        print(f"  {label:28s} {t['prompts']} prompt {t['forward_s']:6.2f}s ({t['prompts_per_s']:5.0f}/s) "
              f"picco {t['peak_vram_gb']:.2f}GB righe<= {t['max_batch_rows']} padding {t['padding_ratio']:.0%} "
              f"oom {t['oom_retries']} [{t['prefix_mode']}, -{t['tokens_saved']} token]", flush=True)

    for budget in budgets:
        t = hot(engine, qs, batch_tokens=budget, **base)
        out["budgets"][budget] = t
        report(f"budget {budget}", t)
    best = max(out["budgets"], key=lambda b: out["budgets"][b]["prompts_per_s"])
    out["best_budget"] = best
    t = hot(engine, qs, batch_tokens=best, prefix_cache=True, **base)
    out["prefix_cache"] = t
    report(f"budget {best} + prefix cache", t)
    if engine.spec.set_attention:
        t = hot(engine, qs, batch_tokens=best, set_attention=True, **base)
        out["set_attention"] = t
        report(f"budget {best} + set attention", t)
    sq = state_questions()
    for label, kw in (("plain", {}), ("prefix_cache", {"prefix_cache": True})):
        t = hot(engine, sq, batch_tokens=best, **kw, **base)
        out["state"][label] = t
        report(f"stato lungo, {label}", t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prompt", "eval", "stress"])
    ap.add_argument("--models", nargs="*", default=[m.key for m in MODELS])
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--batch-size", type=int, default=64, help="righe massime per batch; default per i modelli non in BATCH_SIZE")
    ap.add_argument("--stress-budgets", type=int, nargs="*", help="token per batch; default: STRESS_BUDGETS per modello")
    args = ap.parse_args()

    engine = Engine()
    for key in args.models:
        print(f"== {MODELS_BY_KEY[key].label}", flush=True)
        t0 = time.perf_counter()
        info = engine.load(key)
        print(f"  caricato in {info['load_seconds']}s, VRAM {info['vram_gb']} GB", flush=True)
        bs = BATCH_SIZE.get(key, args.batch_size)
        if args.phase == "prompt":
            data = phase_prompt(engine, key, bs)
        elif args.phase == "eval":
            data = phase_eval(engine, key, bs, args.prompt)
        else:
            budgets = args.stress_budgets or STRESS_BUDGETS.get(key, DEFAULT_STRESS_BUDGETS)
            data = phase_stress(engine, key, args.prompt, budgets)
        save(f"{args.phase}-{key}", {"model": info, "elapsed_s": time.perf_counter() - t0, **{"data": data}})


if __name__ == "__main__":
    main()
