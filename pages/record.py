"""Records real model answers for the static demo (GitHub Pages): examples, test questions and the "Come
funziona" question, for every model, with 10 permutations per question. Writes pages/data/."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import DEFAULT_PROMPT, MODELS, NONE_OPTION, Engine, Question  # noqa: E402

OUT = ROOT / "pages" / "data"
K = 10


def key(q: Question) -> str:
    """Same key as pages/demo.js: type, state, question, options, none option."""
    return json.dumps([q.type, q.state or "", q.question, q.options, q.none_option], ensure_ascii=False,
                      separators=(",", ":"))


def example_questions() -> list[Question]:
    out = []
    for ex in json.loads((ROOT / "data" / "examples.json").read_text()):
        for q in ex["questions"]:
            options = [] if q["type"] == "yesno" else [o.strip() for o in q["options"] if o.strip()]
            none = q["type"] == "choice" and q.get("none", True)
            out.append(Question(q["question"].strip(), options, None, q["type"], none, ex["state"].strip() or None))
    return out


def explain_questions() -> list[Question]:
    html = (ROOT / "static" / "index.html").read_text()
    question = re.search(r'id="xq" value="([^"]*)"', html).group(1).replace("&#39;", "'")
    options = [o.strip() for o in re.search(r'id="xo" value="([^"]*)"', html).group(1).split(",")]
    return [Question(question, options, None, "choice", none) for none in (True, False)]


def test_questions() -> list[Question]:
    raw = json.loads((ROOT / "data" / "test_questions.json").read_text())
    return [Question(q["question"], q["options"], None, "choice", True) for q in raw]


def prompt_template(engine: Engine) -> str:
    return engine.build_prompt(Question("@@Q@@", ["@@O1@@", "@@O2@@"], none_option=False), ["@@O1@@", "@@O2@@"],
                               DEFAULT_PROMPT)


def record(engine: Engine, questions: list[Question]) -> dict:
    variants, timing = engine.run_variants(questions, K, max_questions=None, prefix_cache=True)
    ms = 1000 * timing["forward_s"] / timing["prompts"]
    r = lambda xs, d=4: [round(x, d) for x in xs]  # noqa: E731
    return {key(q): {"t": round(ms, 2), "v": [[v["order"], r(v["position_probs"]), r(v["probs"]), round(v["letter_mass"], 4),
                                             v["unconstrained_top"]] for v in vs]}
            for q, vs in zip(questions, variants)}


def write_config():
    import server
    cfg = server.config()
    cfg.update(gpu=None, engine={}, calibrations={})
    (OUT / "config.json").write_text(json.dumps(cfg, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[m.key for m in MODELS])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    write_config()
    engine = Engine()
    for model in args.models:
        print(engine.load(model), flush=True)
        answers = {}
        for name, qs in (("esempi", example_questions()), ("come funziona", explain_questions()),
                         ("test", test_questions())):
            # the warm-up run compiles the kernels, the second one gives hot timings
            record(engine, qs)
            answers.update(record(engine, qs))
            print(f"  {name}: {len(qs)} domande", flush=True)
        data = {"prompt": prompt_template(engine), "none_option": NONE_OPTION, "answers": answers}
        path = OUT / f"answers-{model}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
        print(f"  {path.name}: {path.stat().st_size / 1024:.0f} KB", flush=True)


if __name__ == "__main__":
    main()
