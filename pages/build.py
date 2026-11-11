"""Builds the static demo in _site/: the real UI with pages/demo.js in place of the server API and
pages/browser.js for the model that runs in the browser."""
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
    scripts = '\n<script src="browser.js"></script>\n<script src="demo.js"></script>'
    # cross-origin isolation (COOP/COEP from a service worker) lets the browser model run WASM on several threads
    isolation = '<script>window.coi = {coepCredentialless: () => true, quiet: true};</script>\n<script src="coi-serviceworker.js"></script>\n</head>'
    if html.count("</head>") != 1:
        raise SystemExit("static/index.html: </head> non trovato")
    (SITE / "index.html").write_text(html.replace(marker, scripts + marker).replace("</head>", isolation))
    for name in ("browser.js", "demo.js", "coi-serviceworker.js"):
        shutil.copy(ROOT / "pages" / name, SITE / name)
    for path in (ROOT / "pages" / "data").glob("*.json"):
        shutil.copy(path, SITE / "data" / path.name)
    for name in ("examples.json", "test_questions.json"):
        shutil.copy(ROOT / "data" / name, SITE / "data" / name)
    bench = {phase: {p.stem.removeprefix(f"{phase}-"): json.loads(p.read_text())
                     for p in sorted(RESULTS.glob(f"{phase}-*.json"))} for phase in ("eval", "stress", "compare")}
    (SITE / "data" / "bench.json").write_text(json.dumps(bench, ensure_ascii=False))
    print(f"{SITE}: {sum(f.stat().st_size for f in SITE.rglob('*') if f.is_file()) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
