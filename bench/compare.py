"""Comparison with open-source decision models and zero-shot classifiers on the same questions.

Systems:
  llm       our models, letters constrained on the next token, with and without scramble
  laya      Convai Laya multilingual (mmBERT, typed decisions in one forward pass)
  nli       MoritzLaurer bge-m3-zeroshot-v2.0 (NLI cross-encoder, one forward pass per option)
  gliclass  Knowledgator GLiClass x-base (mDeBERTa, all options in one forward pass)
  jev       TypeSafe Jev through the classifier.dev API (/v1/systemone), key CLASSIFIER_DEV_API in .env

Every system returns one probability vector per question over Question.all_options. Metrics, temperature
scaling and conformal calibration (fitted on the dev split) are the same for all of them.
"""
import argparse
import gc
import gzip
import json
import math
import random
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from core import LETTERS, MODELS, MODELS_BY_KEY, Engine, Question, build_report, calibrate, true_probs  # noqa: E402
from run_bench import BATCH_SIZE, EPSILON, RESULTS, SCRAMBLE, load_split, local_questions, save  # noqa: E402

DATA = ROOT / "bench" / "data"
SAMPLE = {"massive": {"test": 500, "dev": 200}, "sentiment": {"test": 500, "dev": 300}}
MASSIVE_OPTIONS = 5
BASELINES = {
    "laya": {"label": "Laya multilingual", "repo": "convaiinnovations/laya", "params_b": 0.32},
    "nli": {"label": "bge-m3 zeroshot v2.0", "repo": "MoritzLaurer/bge-m3-zeroshot-v2.0", "params_b": 0.57},
    "gliclass": {"label": "GLiClass x-base", "repo": "knowledgator/gliclass-x-base", "params_b": 0.30},
    "jev": {"label": "Jev (API)", "repo": "classifier.dev, jev-latest", "params_b": None, "remote": True},
}
JEV_URL = "https://classifier.dev/v1/systemone"
JEV_PAUSE_S = 0.15
JEV_CACHE = DATA / "jev_cache.json"

SIB_LABELS = {
    "science/technology": "Scienza e tecnologia", "travel": "Viaggi", "politics": "Politica", "sports": "Sport",
    "health": "Salute", "entertainment": "Intrattenimento", "geography": "Geografia",
}
SENTIMENT_LEVELS = ["Negativo", "Neutro", "Positivo"]
MASSIVE_LABELS = {
    "alarm_query": "Chiedere quali sveglie sono impostate", "alarm_remove": "Cancellare una sveglia",
    "alarm_set": "Impostare una sveglia", "audio_volume_down": "Abbassare il volume",
    "audio_volume_mute": "Silenziare l'audio", "audio_volume_other": "Altre regolazioni del volume",
    "audio_volume_up": "Alzare il volume", "calendar_query": "Consultare il calendario",
    "calendar_remove": "Cancellare un evento dal calendario", "calendar_set": "Aggiungere un evento al calendario",
    "cooking_recipe": "Chiedere una ricetta", "datetime_convert": "Convertire orari o fusi orari",
    "datetime_query": "Chiedere data o ora", "email_addcontact": "Aggiungere un contatto email",
    "email_query": "Controllare le email", "email_querycontact": "Cercare informazioni su un contatto",
    "email_sendemail": "Inviare una email", "general_greet": "Salutare", "general_joke": "Chiedere una barzelletta",
    "iot_cleaning": "Avviare il robot aspirapolvere", "iot_coffee": "Preparare il caffè",
    "iot_hue_lightchange": "Cambiare il colore delle luci", "iot_hue_lightdim": "Abbassare l'intensità delle luci",
    "iot_hue_lightoff": "Spegnere le luci", "iot_hue_lighton": "Accendere le luci",
    "iot_hue_lightup": "Aumentare l'intensità delle luci", "iot_wemo_off": "Spegnere una presa smart",
    "iot_wemo_on": "Accendere una presa smart", "lists_createoradd": "Creare una lista o aggiungervi elementi",
    "lists_query": "Consultare una lista", "lists_remove": "Togliere elementi da una lista",
    "music_dislikeness": "Dire che una canzone non piace", "music_likeness": "Dire che una canzone piace",
    "music_query": "Chiedere informazioni sulla musica in riproduzione", "music_settings": "Cambiare le impostazioni di riproduzione",
    "news_query": "Chiedere le notizie", "play_audiobook": "Riprodurre un audiolibro", "play_game": "Avviare un gioco",
    "play_music": "Riprodurre musica", "play_podcasts": "Riprodurre un podcast", "play_radio": "Ascoltare la radio",
    "qa_currency": "Chiedere un cambio valuta", "qa_definition": "Chiedere una definizione",
    "qa_factoid": "Domanda di cultura generale", "qa_maths": "Fare un calcolo", "qa_stock": "Chiedere quotazioni di borsa",
    "recommendation_events": "Chiedere consigli su eventi", "recommendation_locations": "Chiedere consigli su luoghi",
    "recommendation_movies": "Chiedere consigli su film", "social_post": "Pubblicare sui social",
    "social_query": "Controllare i social", "takeaway_order": "Ordinare cibo da asporto",
    "takeaway_query": "Informazioni su cibo da asporto", "transport_query": "Informazioni sui trasporti",
    "transport_taxi": "Prenotare un taxi", "transport_ticket": "Prenotare un biglietto",
    "transport_traffic": "Chiedere del traffico", "weather_query": "Chiedere il meteo",
}

QUESTIONS = {
    "massive": "Che cosa chiede l'utente?",
    "sib200": "Di quale argomento parla il testo?",
    "sib200-yesno": "Il testo parla di {}?",
    "sentiment": "Qual è il sentimento del tweet?",
}
NLI_TEMPLATES = {
    "massive": "L'utente vuole: {}.",
    "sib200": "Questo testo parla di {}.",
    "sib200-yesno": "Questo testo parla di {}.",
    "sentiment": "Il sentimento di questo tweet è {}.",
    "local": "La risposta corretta è: {}.",
    "mmlu": "La risposta corretta è: {}.",
}
DATASETS = {
    "massive": "MASSIVE intent (5 opzioni)",
    "sib200": "SIB-200 argomento (7 opzioni)",
    "sib200-yesno": "SIB-200 sì / no",
    "sentiment": "Sentiment tweet (punteggio 3 livelli)",
    "local": "256 domande locali",
    "mmlu": "Global-MMLU-Lite",
}


def sample(rows: list, n: int, seed: str) -> list:
    return random.Random(seed).sample(rows, n) if len(rows) > n else rows


def item(q: Question, text: str | None = None, label: str | None = None) -> dict:
    """text: the input classified by the encoders; label: the dataset label, used for the yes/no topic."""
    return {"question": q, "text": text if text is not None else q.question, "label": label}


def massive_split(split: str) -> list[dict]:
    file = "validation" if split == "dev" else "test"
    with gzip.open(DATA / f"massive_it_{file}.json.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    rows = sample([r for r in rows if r["label"] in MASSIVE_LABELS], SAMPLE["massive"][split], f"massive-{split}")
    rng = random.Random(f"massive-options-{split}")
    labels = sorted(MASSIVE_LABELS)
    out = []
    for r in rows:
        domain = r["label"].split("_")[0]
        near = [lab for lab in labels if lab.startswith(domain + "_") and lab != r["label"]]
        far = [lab for lab in labels if not lab.startswith(domain + "_")]
        picked = rng.sample(near, min(2, len(near)))
        picked += rng.sample(far, MASSIVE_OPTIONS - 1 - len(picked))
        options = picked + [r["label"]]
        rng.shuffle(options)
        q = Question(QUESTIONS["massive"], [MASSIVE_LABELS[o] for o in options], LETTERS[options.index(r["label"])],
                     none_option=False, state=r["text"])
        out.append(item(q, r["text"]))
    return out


def sib_rows(split: str) -> list[tuple[str, str]]:
    lines = (DATA / f"sib200_it_{split}.tsv").read_text().splitlines()[1:]
    return [(cat, text) for _, cat, text in (line.split("\t", 2) for line in lines)]


def sib_split(split: str) -> list[dict]:
    labels = list(SIB_LABELS)
    out = []
    for cat, text in sib_rows(split):
        q = Question(QUESTIONS["sib200"], [SIB_LABELS[lab] for lab in labels], LETTERS[labels.index(cat)],
                     none_option=False, state=text)
        out.append(item(q, text))
    return out


def sib_yesno_split(split: str) -> list[dict]:
    """Half of the texts are asked about their own topic (answer Sì), half about another random topic."""
    rng = random.Random(f"sib-yesno-{split}")
    out = []
    for i, (cat, text) in enumerate(sib_rows(split)):
        topic = cat if i % 2 == 0 else rng.choice([lab for lab in SIB_LABELS if lab != cat])
        name = SIB_LABELS[topic].lower()
        q = Question(QUESTIONS["sib200-yesno"].format(name), answer="A" if topic == cat else "B", type="yesno",
                     state=text)
        out.append(item(q, text, name))
    return out


def sentiment_split(split: str) -> list[dict]:
    file = "validation" if split == "dev" else "test"
    rows = [json.loads(line) for line in (DATA / f"sentiment_it_{file}.jsonl").read_text().splitlines()]
    out = []
    for r in sample(rows, SAMPLE["sentiment"][split], f"sentiment-{split}"):
        q = Question(QUESTIONS["sentiment"], list(SENTIMENT_LEVELS), LETTERS[int(r["label"])], type="score",
                     state=r["text"])
        out.append(item(q, r["text"]))
    return out


def local_split(split: str) -> list[dict]:
    return [item(q) for q in local_questions()] if split == "test" else []


def mmlu_split(split: str) -> list[dict]:
    return [item(i["question"]) for i in load_split(split)]


LOADERS = {
    "massive": massive_split, "sib200": sib_split, "sib200-yesno": sib_yesno_split, "sentiment": sentiment_split,
    "local": local_split, "mmlu": mmlu_split,
}


def load(name: str) -> tuple[list[dict], list[dict]]:
    return LOADERS[name]("dev"), LOADERS[name]("test")


def apply_temperature(probs: list[float], t: float) -> list[float]:
    logp = torch.tensor(probs, dtype=torch.float64).clamp_min(1e-12).log() / t
    return logp.softmax(-1).tolist()


def fit_temperature(probs: list[list[float]], answers: list[str]) -> float:
    """Single temperature that minimises the NLL of the correct option (log-spaced grid, 0.05 to 20)."""
    def nll(t):
        return -sum(math.log(max(apply_temperature(p, t)[LETTERS.index(a)], 1e-12)) for p, a in zip(probs, answers))
    grid = [math.exp(x / 20) for x in range(-60, 61)]
    return min(grid, key=nll)


def as_variants(probs: list[list[float]]) -> list[list[dict]]:
    return [[{"probs": p, "letter_mass": 1.0}] for p in probs]


def evaluate(dev: list[dict], test: list[dict], dev_probs, test_probs, aggregation: str = "mean") -> dict:
    """dev_probs / test_probs: per question either a probability vector or a list of variants (LLM)."""
    def variants(probs):
        return probs if probs and isinstance(probs[0][0], dict) else as_variants(probs)
    test_q = [i["question"] for i in test]
    out = {}
    cal = None
    if dev:
        dev_q = [i["question"] for i in dev]
        dev_rep = build_report(dev_q, variants(dev_probs), aggregation)
        cal = calibrate(true_probs(dev_rep), EPSILON, "compare")
        t = fit_temperature([r["probs"] for r in dev_rep["results"]], [q.answer for q in dev_q])
    rep = build_report(test_q, variants(test_probs), aggregation, cal)
    s = rep["summary"]
    out.update({k: s[k] for k in ("accuracy", "ece", "brier", "nll", "reliability", "avg_confidence")})
    if cal:
        out["conformal"] = s["conformal"]
        scaled = [apply_temperature(r["probs"], t) for r in rep["results"]]
        srep = build_report(test_q, as_variants(scaled), "mean")["summary"]
        out["temperature"] = t
        out.update({f"{k}_t": srep[k] for k in ("ece", "brier", "nll")})
    return out


def timing(questions: int, seconds: float, peak_gb: float) -> dict:
    return {"questions": questions, "forward_s": seconds, "ms_per_question": 1000 * seconds / questions,
            "questions_per_s": questions / seconds, "peak_vram_gb": peak_gb}


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30


class LayaRunner:
    def __init__(self, device: str):
        from laya import Agent
        self.agent = Agent(BASELINES["laya"]["repo"], subfolder="multilingual", device=device)

    @staticmethod
    def question_def(q: Question) -> dict:
        if q.type == "yesno":
            return {"type": "noul", "instructions": q.question}
        if q.type == "score":
            return {"type": "score", "instructions": q.question, "criteria": list(q.options)}
        return {"type": "choice", "instructions": q.question,
                "criteria": {LETTERS[i]: o for i, o in enumerate(q.all_options)}}

    @staticmethod
    def probs(q: Question, answer: dict) -> list[float]:
        if q.type == "yesno":
            return [answer["noul"], 1 - answer["noul"]]
        p = answer["probabilities"]
        keys = [str(i) for i in range(len(q.options))] if q.type == "score" else LETTERS[:len(q.all_options)]
        return [p[k] for k in keys]

    def __call__(self, items: list[dict]) -> list[list[float]]:
        """Questions shared by several states go through predict_batch, the rest one state per call."""
        groups: dict[str, tuple[dict, list[int]]] = {}
        for i, it in enumerate(items):
            qdef = self.question_def(it["question"])
            groups.setdefault(json.dumps(qdef, ensure_ascii=False), (qdef, []))[1].append(i)
        out: list = [None] * len(items)
        for qd, idx in groups.values():
            qdef = {"q": qd}
            states = [items[i]["question"].state or items[i]["question"].question for i in idx]
            results = self.agent.predict_batch(states, qdef, batch_size=32)
            for i, r in zip(idx, results):
                out[i] = self.probs(items[i]["question"], r["answers"]["q"])
        return out


class NliRunner:
    def __init__(self, device: str):
        from transformers import pipeline
        self.pipe = pipeline("zero-shot-classification", model=BASELINES["nli"]["repo"], device=device,
                             dtype=torch.float16 if device == "cuda" else torch.float32)

    def __call__(self, items: list[dict], dataset: str) -> list[list[float]]:
        template = NLI_TEMPLATES[dataset]
        out = []
        for it in items:
            q = it["question"]
            if q.type == "yesno":
                r = self.pipe(it["text"], [it["label"]], hypothesis_template=template, multi_label=True)
                out.append([r["scores"][0], 1 - r["scores"][0]])
                continue
            labels = list(dict.fromkeys(q.all_options))
            r = self.pipe(it["text"], labels, hypothesis_template=template, multi_label=False, batch_size=16)
            score = dict(zip(r["labels"], r["scores"]))
            p = [score[o] for o in q.all_options]
            out.append([v / sum(p) for v in p])
        return out


class GliclassRunner:
    def __init__(self, device: str):
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer
        repo = BASELINES["gliclass"]["repo"]
        model = GLiClassModel.from_pretrained(repo)
        tokenizer = AutoTokenizer.from_pretrained(repo, add_prefix_space=True)
        self.pipe = ZeroShotClassificationPipeline(model, tokenizer, classification_type="multi-label",
                                                   device="cuda:0" if device == "cuda" else "cpu",
                                                   progress_bar=False)

    def __call__(self, items: list[dict]) -> list[list[float]]:
        """Sigmoid scores turned back into logits and softmaxed over the options; yes/no keeps the sigmoid."""
        texts, labels, prompts = [], [], []
        for it in items:
            q = it["question"]
            texts.append(it["text"])
            labels.append([it["label"]] if q.type == "yesno" else q.all_options)
            prompts.append(q.question if q.state else "Scegli la risposta corretta alla domanda.")
        results = self.pipe(texts, labels, threshold=0.0, batch_size=16, prompt=prompts)
        out = []
        for it, labs, res in zip(items, labels, results):
            score = {r["label"]: min(max(r["score"], 1e-6), 1 - 1e-6) for r in res}
            if it["question"].type == "yesno":
                out.append([score[labs[0]], 1 - score[labs[0]]])
                continue
            logits = torch.tensor([math.log(score[lab] / (1 - score[lab])) for lab in labs])
            out.append(logits.softmax(-1).tolist())
        return out


def retry_after(value: str | None, default: float) -> float:
    """Retry-After is either seconds or an HTTP date."""
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        from email.utils import parsedate_to_datetime
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return default


def env_key(name: str) -> str:
    if name in os.environ:
        return os.environ[name]
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip('"')
    raise SystemExit(f"{name} mancante in .env")


class JevRunner:
    """One request per question (systemone takes one state), cached on disk with the measured latency.
    The answer shapes are the ones Laya copies, so the question mapping is shared."""
    remote = True

    def __init__(self, device: str):
        self.key = env_key("CLASSIFIER_DEV_API")
        self.cache = json.loads(JEV_CACHE.read_text()) if JEV_CACHE.exists() else {}
        self.model = None

    def request(self, body: dict) -> tuple[dict, float]:
        data = json.dumps({**body, "share_data": False}).encode()
        for attempt in range(8):
            req = urllib.request.Request(JEV_URL, data, {
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                "X-Classifier-Share-Data": "false", "User-Agent": "decisionllm-bench"})
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.loads(r.read()), 1000 * (time.perf_counter() - t0)
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504):
                    raise SystemExit(f"classifier.dev {e.code}: {e.read()[:300]!r}")
                time.sleep(retry_after(e.headers.get("Retry-After"), 2 ** attempt))
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2 ** attempt)
        raise SystemExit("classifier.dev: troppi tentativi falliti")

    def answer(self, q: Question) -> dict:
        body = {"model": "jev-latest", "state": q.state or q.question, "questions": {"q": LayaRunner.question_def(q)}}
        key = json.dumps(body, ensure_ascii=False, sort_keys=True)
        if key not in self.cache:
            resp, ms = self.request(body)
            self.cache[key] = {"answer": resp["answers"]["q"], "model": resp.get("model"), "ms": ms}
            if len(self.cache) % 50 == 0:
                self.save()
            time.sleep(JEV_PAUSE_S)
        self.model = self.cache[key]["model"] or self.model
        return self.cache[key]

    def save(self):
        JEV_CACHE.write_text(json.dumps(self.cache, ensure_ascii=False))

    def __call__(self, items: list[dict]) -> list[list[float]]:
        out = []
        for it in items:
            p = LayaRunner.probs(it["question"], self.answer(it["question"])["answer"])
            total = sum(p)
            out.append([v / total for v in p] if total > 0 else [1 / len(p)] * len(p))
        self.save()
        return out

    def ms(self, items: list[dict]) -> float:
        return sum(self.answer(it["question"])["ms"] for it in items) / 1000


def run_baseline(name: str, datasets: list[str], device: str):
    reset_peak()
    t0 = time.perf_counter()
    runner = {"laya": LayaRunner, "nli": NliRunner, "gliclass": GliclassRunner, "jev": JevRunner}[name](device)
    remote = getattr(runner, "remote", False)
    info = {**BASELINES[name], "key": name, "device": device, "load_seconds": round(time.perf_counter() - t0, 2)}
    call = (lambda items, ds: runner(items, ds)) if name == "nli" else (lambda items, ds: runner(items))
    for ds in datasets:
        dev, test = load(ds)
        if not remote:
            call(test[:32], ds)
        dev_probs = call(dev, ds) if dev else []
        reset_peak()
        t0 = time.perf_counter()
        test_probs = call(test, ds)
        elapsed = runner.ms(test) if remote else time.perf_counter() - t0
        entry = {"single": evaluate(dev, test, dev_probs, test_probs),
                 "timing": timing(len(test), elapsed, 0.0 if remote else peak()), "dev_questions": len(dev)}
        report(info["label"], ds, entry)
        store(name, {**info, "model": runner.model} if remote else info, ds, entry)


def run_llm(engine: Engine, key: str, datasets: list[str], batch_size: int):
    info = {**engine.load(key), "key": key, "params_b": MODELS_BY_KEY[key].params_b}
    for ds in datasets:
        dev, test = load(ds)
        dev_q, test_q = [i["question"] for i in dev], [i["question"] for i in test]
        kw = {"batch_size": batch_size, "max_questions": None, "prefix_cache": True}
        dev_v = engine.run_variants(dev_q, SCRAMBLE, **kw)[0] if dev else []
        engine.run_variants(test_q[:64], SCRAMBLE, **kw)
        test_v, t5 = engine.run_variants(test_q, SCRAMBLE, **kw)
        engine.run_variants(test_q[:64], 1, **kw)
        test_v1, t1 = engine.run_variants(test_q, 1, **kw)
        entry = {
            "single": evaluate(dev, test, [v[:1] for v in dev_v], test_v1),
            "scramble": evaluate(dev, test, dev_v, test_v, "geomean"),
            "timing": timing(len(test), t1["total_s"], t1["peak_vram_gb"]),
            "timing_scramble": timing(len(test), t5["total_s"], t5["peak_vram_gb"]),
            "dev_questions": len(dev),
        }
        report(info["label"], ds, entry)
        store(key, info, ds, entry)


def report(label: str, ds: str, r: dict):
    for mode in ("single", "scramble"):
        if mode in r:
            m, t = r[mode], r["timing" if mode == "single" else "timing_scramble"]
            extra = f" ece_t={m['ece_t']:.3f} T={m['temperature']:.2f}" if "ece_t" in m else ""
            print(f"  {label:24s} {ds:13s} {mode:8s} acc={m['accuracy']:.3f} ece={m['ece']:.3f}{extra} "
                  f"brier={m['brier']:.3f} {t['ms_per_question']:.1f} ms/domanda", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=["llm", "baselines"])
    ap.add_argument("--models", nargs="*", help="llm: chiavi dei modelli; baselines: laya nli gliclass jev")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--fresh", action="store_true", help="ricalcola anche i dataset già salvati")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.system == "baselines":
        for name in args.models or list(BASELINES):
            if todo := pending(name, args.datasets, args.fresh):
                run_baseline(name, todo, device)
                gc.collect()
                torch.cuda.empty_cache()
        return
    engine = Engine()
    for key in args.models or [m.key for m in MODELS]:
        if todo := pending(key, args.datasets, args.fresh):
            run_llm(engine, key, todo, BATCH_SIZE.get(key, args.batch_size))


def pending(key: str, datasets: list[str], fresh: bool) -> list[str]:
    """Datasets still to run: a restarted run resumes after the last saved dataset."""
    path = RESULTS / f"compare-{key}.json"
    if fresh:
        path.unlink(missing_ok=True)
    if not path.exists():
        return list(datasets)
    done = json.loads(path.read_text())["datasets"]
    skipped = [d for d in datasets if d in done]
    if skipped:
        print(f"  {key}: già salvati {', '.join(skipped)}", flush=True)
    return [d for d in datasets if d not in done]


def store(key: str, system: dict, dataset: str, entry: dict):
    """Saves one dataset as soon as it is done; the other datasets in the file are kept."""
    path = RESULTS / f"compare-{key}.json"
    datasets = json.loads(path.read_text())["datasets"] if path.exists() else {}
    save(f"compare-{key}", {"system": system, "datasets": {**datasets, dataset: entry}})


if __name__ == "__main__":
    main()
