"""Builds the static demo in _site/: the real UI with pages/demo.js in place of the server API."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "_site"
RESULTS = ROOT / "bench" / "results"


def main():
    shutil.rmtree(SITE, ignore_errors=True)
    (SITE / "data").mkdir(parents=True)
    html = (ROOT / "static" / "index.html").read_text()
    marker = "\n<script>\n"
    if html.count(marker) != 1:
        raise SystemExit("static/index.html: script principale non trovato")
    (SITE / "index.html").write_text(html.replace(marker, '\n<script src="demo.js"></script>' + marker))
    shutil.copy(ROOT / "pages" / "demo.js", SITE / "demo.js")
    for path in (ROOT / "pages" / "data").glob("*.json"):
        shutil.copy(path, SITE / "data" / path.name)
    for name in ("examples.json", "test_questions.json"):
        shutil.copy(ROOT / "data" / name, SITE / "data" / name)
    bench = {phase: {p.stem.removeprefix(f"{phase}-"): json.loads(p.read_text())
                     for p in sorted(RESULTS.glob(f"{phase}-*.json"))} for phase in ("eval", "stress")}
    (SITE / "data" / "bench.json").write_text(json.dumps(bench, ensure_ascii=False))
    print(f"{SITE}: {sum(f.stat().st_size for f in SITE.rglob('*') if f.is_file()) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
