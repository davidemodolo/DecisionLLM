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


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def write_md(ev, st):
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
            rows = [(f"budget {b} token", v) for b, v in d["budgets"].items()]
            rows.append((f"budget {d['best_budget']} + cache prefisso", d["prefix_cache"]))
            if "set_attention" in d:
                rows.append((f"budget {d['best_budget']} + set attention", d["set_attention"]))
            for label, v in rows:
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
    write_md(ev, st)
    print(f"grafici in {PLOTS}, report in bench/RESULTS.md ({len(ev)} modelli valutati, {len(st)} stress)")


if __name__ == "__main__":
    main()
