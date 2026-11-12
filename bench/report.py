"""Builds bench/plots/*.png and bench/RESULTS.md from bench/results/*.json."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import MODELS  # noqa: E402

sys.path.insert(0, str(ROOT / "bench"))
from compare import BASELINES, DATASETS  # noqa: E402
from run_bench import STATE_QUESTIONS as STATE_N  # noqa: E402

RESULTS = ROOT / "bench" / "results"
PLOTS = ROOT / "bench" / "plots"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CONFIGS = [("s1-single", "Senza scramble"), ("s5-geomean", "Scramble ×5, media geometrica")]
CHANCE = 0.8 * 1 / 5 + 0.2 * 1 / 4

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
    "legend.frameon": False, "axes.titleweight": "bold", "axes.titlesize": 12,
})


def load(prefix: str) -> dict:
    out = {}
    for m in MODELS:
        f = RESULTS / f"{prefix}-{m.key}.json"
        if f.exists():
            out[m.key] = json.loads(f.read_text())
    return out


def short(key: str) -> str:
    return next(m.label for m in MODELS if m.key == key).replace(" (4-bit)", "").replace(" Instruct", "")


def params(key: str) -> float:
    return next(m.params_b for m in MODELS if m.key == key)


def family(key: str) -> str:
    return key.split("-")[0]


def scatter_by_params(ax, ev, metric, configs=CONFIGS, pct=True):
    keys = sorted(ev, key=params)
    for ci, (cfg, label) in enumerate(configs):
        color = SERIES[ci]
        xs = [params(k) for k in keys]
        ys = [ev[k]["data"]["configs"][cfg][metric] for k in keys]
        qwen = [(x, y) for k, x, y in zip(keys, xs, ys) if family(k) == "qwen3.5"]
        if len(qwen) > 1:
            ax.plot(*zip(*qwen), color=color, linewidth=2, zorder=2)
        ax.scatter(xs, ys, s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=3, label=label)
    for k in keys:
        y = max(ev[k]["data"]["configs"][c][metric] for c, _ in configs)
        ax.annotate(short(k), (params(k), y), textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=9, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1B", "2B", "4B", "8B"])
    ax.set_xlabel("parametri (scala log)")
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def plot_accuracy(ev):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    scatter_by_params(ax, ev, "accuracy")
    ax.axhline(CHANCE, color=MUTED, linestyle=(0, (4, 4)), linewidth=1)
    ax.text(0.8, CHANCE + 0.006, "caso", color=MUTED, fontsize=9)
    ax.set_ylim(0.15, None)
    ax.set_ylabel("accuratezza")
    ax.set_title("Accuratezza vs parametri (Global-MMLU-Lite IT, 400 domande)", loc="left")
    ax.legend(loc="upper left")
    fig.text(0.01, 0.01, "Linea: famiglia Qwen3.5. Il 20% delle domande ha \"Nessuna delle altre\" come risposta corretta.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(PLOTS / "accuracy_vs_params.png", dpi=150)
    plt.close(fig)


def plot_calibration(ev):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, metric, title in ((axes[0], "ece", "ECE (più basso è meglio)"), (axes[1], "brier", "Brier (più basso è meglio)")):
        scatter_by_params(ax, ev, metric, pct=False)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, None)
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS / "calibration_vs_params.png", dpi=150)
    plt.close(fig)


def plot_conformal(ev):
    cfg = [("s5-geomean", "Scramble ×5, media geometrica")]
    keys = sorted(ev, key=params)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, field, title in ((axes[0], "decided_rate", "Domande decise da sole (set di 1)"),
                             (axes[1], "accuracy_when_decided", "Accuratezza quando decide")):
        xs = [params(k) for k in keys]
        ys = [ev[k]["data"]["configs"][cfg[0][0]]["conformal"][field] or 0 for k in keys]
        ax.scatter(xs, ys, s=64, color=SERIES[1], edgecolor=SURFACE, linewidth=2, zorder=3)
        qwen = [(x, y) for k, x, y in zip(keys, xs, ys) if family(k) == "qwen3.5"]
        ax.plot(*zip(*qwen), color=SERIES[1], linewidth=2, zorder=2)
        for k, x, y in zip(keys, xs, ys):
            ax.annotate(short(k), (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=MUTED)
        ax.set_xscale("log"); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1B", "2B", "4B", "8B"])
        ax.set_xlabel("parametri (scala log)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left")
    eps = next(iter(ev.values()))["data"]["epsilon"]
    fig.suptitle(f"Conformal prediction, ε = {eps}: calibrata su dev, misurata su test", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS / "conformal_vs_params.png", dpi=150)
    plt.close(fig)


def plot_reliability(ev):
    keys = sorted(ev, key=params)
    cols = 3
    rows = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.6 * rows), squeeze=False)
    for ax, k in zip(axes.flat, keys):
        rel = ev[k]["data"]["configs"]["s5-geomean"]["reliability"]
        ax.plot([0, 1], [0, 1], color=MUTED, linestyle=(0, (4, 4)), linewidth=1)
        ax.bar([r["lo"] for r in rel], [r["accuracy"] for r in rel], width=0.1 - 0.008, align="edge",
               color=SERIES[0], zorder=2)
        ax.scatter([r["confidence"] for r in rel], [r["accuracy"] for r in rel], s=16, color=INK, zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ece = ev[k]["data"]["configs"]["s5-geomean"]["ece"]
        ax.set_title(f"{short(k)}  ·  ECE {ece:.3f}", loc="left", fontsize=11)
        ax.set_xlabel("confidenza"); ax.set_ylabel("accuratezza")
    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)
    fig.suptitle("Reliability diagram (scramble ×5, media geometrica): barre = accuratezza per fascia di confidenza",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS / "reliability.png", dpi=150)
    plt.close(fig)


def grouped_bars(ax, keys, series, fmt):
    width = 0.8 / len(series)
    for si, (label, values) in enumerate(series):
        xs = [i + (si - (len(series) - 1) / 2) * width for i in range(len(keys))]
        ax.bar(xs, values, width=width - 0.03, color=SERIES[si], label=label, zorder=2)
        for x, v in zip(xs, values):
            ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([short(k) for k in keys], rotation=20, ha="right")
    ax.grid(axis="x", visible=False)


def plot_speed(st):
    keys = sorted(st, key=params)
    best = {k: st[k]["data"]["budgets"][str(st[k]["data"]["best_budget"])] for k in keys}
    cache = {k: st[k]["data"]["prefix_cache"] for k in keys}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    grouped_bars(axes[0], keys, [("Senza cache", [best[k]["prompts_per_s"] for k in keys]),
                                 ("Cache del prefisso", [cache[k]["prompts_per_s"] for k in keys])], "{:.0f}")
    axes[0].set_title("Prompt/s, 128 domande × 5 permutazioni (a caldo)", loc="left")
    axes[0].legend(loc="upper right")
    grouped_bars(axes[1], keys, [("Senza cache", [best[k]["peak_vram_gb"] for k in keys]),
                                 ("Cache del prefisso", [cache[k]["peak_vram_gb"] for k in keys])], "{:.1f}")
    axes[1].axhline(8, color=MUTED, linestyle=(0, (4, 4)), linewidth=1)
    axes[1].text(len(keys) - 0.5, 8.1, "8 GB GPU", ha="right", fontsize=9, color=MUTED)
    axes[1].set_title("Picco VRAM (GB)", loc="left")
    fig.tight_layout()
    fig.savefig(PLOTS / "speed.png", dpi=150)
    plt.close(fig)

    state = {k: st[k]["data"]["state"] for k in keys}
    fig, ax = plt.subplots(figsize=(8, 4.4))
    per_q = lambda t: 1000 * t["forward_s"] / len(STATE_N)  # noqa: E731
    grouped_bars(ax, keys, [("Senza cache", [per_q(state[k]["plain"]) for k in keys]),
                            ("Cache del prefisso", [per_q(state[k]["prefix_cache"]) for k in keys])], "{:.0f}")
    ax.set_title("Stato lungo + 20 domande × 5 permutazioni: ms per domanda", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS / "state_prefix_cache.png", dpi=150)
    plt.close(fig)
    return best


CLASSIFICATION = ["massive", "sib200", "sib200-yesno", "sentiment"]


def load_compare() -> list[dict]:
    """One row per system and mode: baselines (single pass) and our LLMs with 1 pass and 5 permutations."""
    rows = []
    for key, base in BASELINES.items():
        f = RESULTS / f"compare-{key}.json"
        if f.exists():
            rows.append({"label": base["label"], "mode": "API" if base.get("remote") else "1 passata", "kind": "base",
                         "params": base["params_b"],
                         "datasets": {d: (r["single"], {**r["timing"], "dev_questions": r.get("dev_questions")})
                                      for d, r in json.loads(f.read_text())["datasets"].items()}})
    for m in MODELS:
        f = RESULTS / f"compare-{m.key}.json"
        if not f.exists():
            continue
        ds = json.loads(f.read_text())["datasets"]
        for mode, metrics, timing in (("1 passata", "single", "timing"), ("5 permutazioni", "scramble", "timing_scramble")):
            rows.append({"label": short(m.key), "mode": mode, "kind": "llm1" if metrics == "single" else "llm5",
                         "params": m.params_b, "key": m.key,
                         "datasets": {d: (r[metrics], {**r[timing], "dev_questions": r.get("dev_questions")})
                                      for d, r in ds.items()}})
    return rows


def avg(row: dict, f, datasets=CLASSIFICATION):
    vals = [f(*row["datasets"][d]) for d in datasets if d in row["datasets"]]
    return sum(vals) / len(vals) if len(vals) == len(datasets) and None not in vals else None


def plot_compare(rows):
    fig, ax = plt.subplots(figsize=(8, 5))
    points = []
    for r in rows:
        x, y = avg(r, lambda m, t: t["ms_per_question"]), avg(r, lambda m, t: m["accuracy"])
        if x is None:
            continue
        style = {"base": dict(marker="D", color=SERIES[1], s=70), "llm1": dict(marker="o", facecolor=SURFACE,
                 edgecolor=MUTED, s=50), "llm5": dict(marker="o", color=SERIES[0], s=70)}[r["kind"]]
        ax.scatter(x, y, zorder=3, **style)
        points.append((r, x, y))
    ax.set_xscale("log")
    fig.canvas.draw()
    taken = [ax.transData.transform((x, y)) for _, x, y in points]
    for r, x, y in points:
        if r["kind"] == "llm1":
            continue
        px, py = ax.transData.transform((x, y))
        width = 6.5 * len(r["label"])
        for dx, dy in ((7, 3), (-7, 3), (7, -14), (-7, -14), (7, 14), (-7, 14), (7, -26)):
            left = px + dx if dx > 0 else px + dx - width
            box = (left, py + dy - 4, left + width, py + dy + 10)
            if not any(box[0] - 6 < tx < box[2] + 6 and box[1] - 6 < ty < box[3] + 6 for tx, ty in taken):
                break
        taken += [(bx, box[1] + 7) for bx in range(int(box[0]), int(box[2]), 8)]
        ax.annotate(r["label"], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=9,
                    ha="left" if dx > 0 else "right")
    for key in {r.get("key") for r in rows if r.get("key")}:
        pair = [r for r in rows if r.get("key") == key]
        pts = [(avg(r, lambda m, t: t["ms_per_question"]), avg(r, lambda m, t: m["accuracy"])) for r in pair]
        if len(pts) == 2 and None not in (pts[0][0], pts[1][0]):
            ax.plot(*zip(*pts), color=GRID, linewidth=2, zorder=2)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.set_xlabel("ms per domanda, a caldo (scala logaritmica)")
    ax.set_ylabel("accuratezza media")
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_title("Classificazione in italiano: accuratezza e velocità")
    ax.scatter([], [], marker="D", color=SERIES[1], label="decision model")
    ax.scatter([], [], marker="o", facecolor=SURFACE, edgecolor=MUTED, label="LLM, 1 passata")
    ax.scatter([], [], marker="o", color=SERIES[0], label="LLM, 5 permutazioni")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS / "compare_accuracy_latency.png", dpi=150)
    plt.close(fig)


def compare_md(rows) -> list[str]:
    datasets = [d for d in DATASETS if any(d in r["datasets"] for r in rows)]
    L = ["", "## Confronto con i decision model", "",
         "Gli stessi dataset per tutti i sistemi, con i modelli locali sulla stessa GPU. Decision model: Laya multilingual (Convai, mmBERT "
         "322M, stato e domande tipizzate), Clef-flash (Cloudflare, Qwen3.5 9B con una testa sulle opzioni, in 4 bit), bge-m3 zeroshot v2.0 (MoritzLaurer, cross-encoder NLI, una passata per "
         "opzione), GLiClass x-base (Knowledgator, mDeBERTa, tutte le opzioni in una passata), Jev di TypeSafe via "
         "l'API di classifier.dev e Gemini Decision 1 via la Gemini Decisions API (i tempi dei due servizi API "
         "includono la rete). Ogni sistema "
         "restituisce una distribuzione sulle opzioni della domanda. Sul dev si fittano una temperatura (ECE con T) e "
         "la soglia della conformal prediction (ε 10%).", "",
         "Rigenerare con `uv sync --group baselines && uv run python bench/compare.py baselines && "
         "uv run python bench/compare.py llm`.", "",
         "| Dataset | Test | Dev | Tipo |", "|---|---|---|---|"]
    kinds = {"massive": "scelta tra 5 intent, fino a 2 dello stesso dominio", "sib200": "scelta tra 7 argomenti",
             "sib200-yesno": "\"Il testo parla di X?\", metà vere", "sentiment": "punteggio su 3 livelli",
             "local": "scelta tra 4 opzioni più \"Nessuna delle altre\"",
             "mmlu": "come sopra, 20% con risposta \"Nessuna delle altre\""}
    for d in datasets:
        m, t = next(r["datasets"][d] for r in rows if d in r["datasets"])
        L.append(f"| {DATASETS[d]} | {t['questions']} | {t.get('dev_questions') or '–'} | {kinds[d]} |")
    L += ["", "![Confronto](plots/compare_accuracy_latency.png)", "", "### Accuratezza per dataset", "",
         "| Sistema | Modalità | Parametri | " + " | ".join(DATASETS[d] for d in datasets) + " | Media classificazione |",
         "|---|---|---|" + "---|" * (len(datasets) + 1)]
    for r in rows:
        cells = [pct(r["datasets"][d][0]["accuracy"]) if d in r["datasets"] else "–" for d in datasets]
        L.append(f"| {r['label']} | {r['mode']} | {str(r['params']) + 'B' if r['params'] else '–'} | " + " | ".join(cells) +
                 f" | **{pct(avg(r, lambda m, t: m['accuracy']))}** |")
    L += ["", "### Calibrazione, conformal e velocità (media dei 4 dataset di classificazione)", "",
          "| Sistema | Modalità | ECE | ECE con T | Brier | Decide da solo | Acc. quando decide | ms / domanda | Picco VRAM |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda key, sub=None: avg(r, lambda m, t: (m.get("conformal") or {}).get(sub) if key == "conformal" else m.get(key))  # noqa: E731
        num = lambda v, d=3: "–" if v is None else f"{v:.{d}f}"  # noqa: E731
        L.append(f"| {r['label']} | {r['mode']} | {num(f('ece'))} | {num(f('ece_t'))} | {num(f('brier'))} | "
                 f"{pct(f('conformal', 'decided_rate'))} | {pct(f('conformal', 'accuracy_when_decided'))} | "
                 f"{num(avg(r, lambda m, t: t['ms_per_question']), 1)} | {'API' if r['mode'] == 'API' else num(avg(r, lambda m, t: t['peak_vram_gb']), 2) + ' GB'} |")
    L += ["", "Note sul metodo:", "",
          "- Laya valuta in un'unica passata più stati con la stessa domanda (`predict_batch`). Su MASSIVE, domande "
          "locali e MMLU le opzioni cambiano da domanda a domanda, quindi Laya fa una chiamata per domanda.",
          "- Il cross-encoder NLI riceve un'ipotesi per opzione (\"Questo testo parla di {}.\", \"L'utente vuole: {}.\"...) "
          "e la probabilità di implicazione normalizzata sulle opzioni. Per il sì / no usa la probabilità di "
          "implicazione dell'ipotesi.",
          "- GLiClass dà un sigmoid per opzione: sulle domande a scelta i logit vengono normalizzati con una softmax, "
          "sul sì / no il sigmoid dell'argomento chiesto è P(sì). Con una sola etichetta il sigmoid resta quasi sempre "
          "sotto 0.5, quindi la risposta è quasi sempre \"No\" e l'accuratezza è vicina al caso. La temperatura "
          "cambia la scala ma non la soglia.",
          "- Jev è chiamato via API (classifier.dev, `/v1/systemone`), una richiesta per domanda: i tempi includono la "
          "rete. L'API arrotonda le probabilità a due decimali, quindi alcune opzioni valgono 0: la temperatura fittata "
          "sul NLL tende a valori alti e l'ECE grezza è la misura più fedele.",
          "- GLiDE, vela3.1-fast e d3-lite girano con il codice dei loro rilasci (`glide`, `vela3_encoder.DecisionEncoder`, "
          "`AutoModel` con `trust_remote_code`) sulla stessa GPU. vela3.1-fast non ha il tipo punteggio: sul sentiment i "
          "tre livelli passano come scelta. d3-lite risponde a una richiesta per chiamata, GLiDE e vela elaborano le "
          "domande in batch. Su MMLU vela3.1-fast resta al livello del caso (25% con quattro opzioni) e sulle domande "
          "locali sbaglia due volte su tre con confidenza alta (ECE 0.28): risponde bene quando le opzioni sono "
          "etichette brevi di intenti o argomenti. d3-lite arriva all'83% sui dataset di classificazione con 0.85B "
          "di parametri e ha l'ECE grezza più bassa della tabella (0.041).",
          "- Clef-flash usa il codice e la testa del rilascio (`joint_schema_model.py`). Il backbone è caricato come il "
          "nostro Qwen3.5 9B (layer in NF4 sulla GPU, embedding e `lm_head` sulla CPU), perché il rilascio in bf16 "
          "richiede circa 19 GB: i risultati possono scostarsi da quelli in bf16. Sul sentiment (domanda a "
          "punteggio) Clef tende al livello centrale: sui primi 200 tweet del test risponde \"Neutro\" al 79% dei "
          "negativi e al 57% dei positivi.",
          "- Gemini Decision 1 è chiamato via Gemini Decisions API (`/v1beta/decisions`, `gemini-decision-1`), "
          "una richiesta per domanda con le stesse domande tipizzate di Jev: i tempi includono la rete. Un tweet "
          "del sentiment viene rifiutato dai filtri di sicurezza (HTTP 400) e conta come probabilità uniformi.",
          "- LFM2.5 1.2B risponde \"Sì\" a ogni domanda sì / no di SIB-200 con probabilità vicina a 1, in entrambi "
          "gli ordini delle opzioni: segue la parola, non la lettera. Accuratezza 50% con e senza permutazioni.",
          "- I nostri LLM con una passata leggono le opzioni nell'ordine originale; con 5 permutazioni le opzioni "
          "ruotano e le probabilità sono aggregate con la media geometrica. La temperatura e la soglia conforme sono "
          "fittate separatamente per ogni modalità."]
    return L


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def write_md(ev, st, rows):
    keys = sorted(ev, key=params)
    L = ["# Risultati del benchmark", "",
         "Dataset: Global-MMLU-Lite, split italiano. 215 domande dev (calibrazione), 400 test (valutazione).",
         "Al 20% delle domande è stata tolta l'opzione corretta, quindi la risposta giusta è \"Nessuna delle altre\".",
         f"Accuratezza del caso: {pct(CHANCE)}. Prompt: `{next(iter(ev.values()))['data']['prompt']}`. GPU: 8 GB.", "",
         "Rigenerare con `uv run python bench/run_bench.py eval && uv run python bench/run_bench.py stress && "
         "uv run python bench/report.py`.", "",
         "## Accuratezza", "", "![Accuratezza vs parametri](plots/accuracy_vs_params.png)", "",
         "| Modello | Parametri | Senza scramble | Scramble geomean | mean | weighted | vote | Su domande \"nessuna\" (geomean) |",
         "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        c = ev[k]["data"]["configs"]
        L.append(f"| {short(k)} | {params(k)}B | {pct(c['s1-single']['accuracy'])} | {pct(c['s5-geomean']['accuracy'])} | "
                 f"{pct(c['s5-mean']['accuracy'])} | {pct(c['s5-weighted']['accuracy'])} | {pct(c['s5-vote']['accuracy'])} | "
                 f"{pct(c['s5-geomean']['accuracy_nota'])} |")
    L += ["", "## Calibrazione", "", "![Calibrazione vs parametri](plots/calibration_vs_params.png)", "",
          "![Reliability](plots/reliability.png)", "",
          "| Modello | ECE senza scramble | ECE scramble | Brier senza scramble | Brier scramble | Massa sulle lettere |",
          "|---|---|---|---|---|---|"]
    for k in keys:
        c = ev[k]["data"]["configs"]
        L.append(f"| {short(k)} | {c['s1-single']['ece']:.3f} | {c['s5-geomean']['ece']:.3f} | "
                 f"{c['s1-single']['brier']:.3f} | {c['s5-geomean']['brier']:.3f} | {pct(c['s1-single']['letter_mass'])} |")
    eps = next(iter(ev.values()))["data"]["epsilon"]
    L += ["", f"## Conformal prediction (ε = {eps})", "", "![Conformal](plots/conformal_vs_params.png)", "",
          "| Modello | Copertura | Decise da sole | Accuratezza quando decide | Set medio |", "|---|---|---|---|---|"]
    for k in keys:
        cf = ev[k]["data"]["configs"]["s5-geomean"]["conformal"]
        L.append(f"| {short(k)} | {pct(cf['coverage'])} | {pct(cf['decided_rate'])} | "
                 f"{pct(cf['accuracy_when_decided'])} | {cf['avg_set_size']:.2f} |")
    set_keys = [k for k in keys if "s1-single-set" in ev[k]["data"]["configs"]]
    if set_keys:
        L += ["", "## Set-based attention (Granite)", "",
              "Le opzioni condividono le stesse posizioni e non si vedono tra loro: il risultato non dipende dal loro "
              "ordine. Le lettere restano legate alle opzioni, quindi il bias sulle etichette resta.", "",
              "| Modello | Config | Accuratezza | ECE | Bias posizionale A-E |", "|---|---|---|---|---|"]
        for k in set_keys:
            for cfg, label in (("s1-single", "normale"), ("s1-single-set", "set attention"),
                               ("s5-geomean", "scramble"), ("s5-geomean-set", "set attention + scramble")):
                c = ev[k]["data"]["configs"][cfg]
                bias = " ".join(f"{100 * b:.0f}" for b in c["position_bias"])
                L.append(f"| {short(k)} | {label} | {pct(c['accuracy'])} | {c['ece']:.3f} | {bias} |")
    if st:
        L += ["", "## Velocità e memoria (a caldo)", "", "![Velocità](plots/speed.png)", "",
              "128 domande di test × 5 permutazioni = 640 prompt. Ogni configurazione è eseguita due volte e si "
              "misura la seconda (i kernel Triton vengono compilati alla prima).", "",
              "| Modello | Config | Forward (s) | Prompt/s | Token/s | Picco VRAM | Righe max | Padding |",
              "|---|---|---|---|---|---|---|---|"]
        for k in sorted(st, key=params):
            d = st[k]["data"]
            speed_rows = [(f"budget {b} token", v) for b, v in d["budgets"].items()]
            speed_rows.append((f"budget {d['best_budget']} + cache prefisso", d["prefix_cache"]))
            if "set_attention" in d:
                speed_rows.append((f"budget {d['best_budget']} + set attention", d["set_attention"]))
            for label, v in speed_rows:
                L.append(f"| {short(k)} | {label} | {v['forward_s']:.1f} | {v['prompts_per_s']:.0f} | "
                         f"{v['tokens_per_s']:.0f} | {v['peak_vram_gb']:.2f} GB | {v['max_batch_rows']} | "
                         f"{100 * v['padding_ratio']:.0f}% |")
        L += ["", "Le domande di test sono brevi e il testo in comune tra le permutazioni è circa il 15% dei token: qui "
              "la copia della cache costa più del prefill risparmiato. Per questo il motore attiva la cache solo quando "
              "risparmia almeno il 25% dei token, come con uno stato condiviso.",
              "", "### Stato lungo condiviso", "", "![Stato](plots/state_prefix_cache.png)", "",
              f"Uno stato di circa 250 parole e {len(STATE_N)} domande (scelta, punteggio, sì/no) × 5 permutazioni.", "",
              "| Modello | Senza cache (ms/domanda) | Con cache (ms/domanda) | Speedup | Token risparmiati |",
              "|---|---|---|---|---|"]
        for k in sorted(st, key=params):
            sd = st[k]["data"]["state"]
            a, b = sd["plain"]["forward_s"], sd["prefix_cache"]["forward_s"]
            L.append(f"| {short(k)} | {1000 * a / len(STATE_N):.0f} | {1000 * b / len(STATE_N):.0f} | {a / b:.1f}× | "
                     f"{sd['prefix_cache']['tokens_saved']} |")
    if rows:
        L += compare_md(rows)
    L += ["", "## Note", "",
          "- In bf16 il risultato di un prompt può variare leggermente con la composizione del batch (forme diverse, "
          "kernel diversi): sulle domande incerte circa l'1-2% degli argmax cambia. In fp32 il padding non altera i "
          "risultati (verificato).",
          "- La GPU di test è una RTX PRO 1000 Laptop (8 GB) limitata in potenza: sotto carico lavora a circa 1 GHz su 3.09."]
    (ROOT / "bench" / "RESULTS.md").write_text("\n".join(L) + "\n")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    ev, st = load("eval"), load("stress")
    plot_accuracy(ev)
    plot_calibration(ev)
    plot_conformal(ev)
    plot_reliability(ev)
    if st:
        plot_speed(st)
    rows = load_compare()
    if rows:
        plot_compare(rows)
    write_md(ev, st, rows)
    print(f"grafici in {PLOTS}, report in bench/RESULTS.md ({len(ev)} modelli valutati, {len(st)} stress)")


if __name__ == "__main__":
    main()
