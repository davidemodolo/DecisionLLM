"""Downloads the model weights (and the Global-MMLU-Lite Italian split) into the Hugging Face cache."""
import argparse
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from core import MODELS, MODELS_BY_KEY

ROOT = Path(__file__).resolve().parent
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "tokenizer*"]
BENCHMARK_FILES = {
    f"global_mmlu_lite_it_{split}.parquet": ("CohereLabs/Global-MMLU-Lite", f"it/{split}-00000-of-00001.parquet")
    for split in ("dev", "test")
} | {
    f"massive_it_{split}.json.gz": ("mteb/amazon_massive_intent", f"{split}/it.json.gz")
    for split in ("validation", "test")
} | {
    f"sib200_it_{split}.tsv": ("Davlan/sib200", f"data/ita_Latn/{split}.tsv") for split in ("dev", "test")
} | {
    f"sentiment_it_{split}.jsonl": ("cardiffnlp/tweet_sentiment_multilingual", f"data/italian/{split}.jsonl")
    for split in ("validation", "test")
}


def download_benchmark():
    target = ROOT / "bench" / "data"
    target.mkdir(parents=True, exist_ok=True)
    for name, (repo, filename) in BENCHMARK_FILES.items():
        dest = target / name
        if not dest.exists():
            dest.write_bytes(Path(hf_hub_download(repo, filename, repo_type="dataset")).read_bytes())
    print(f"benchmark in {target}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=[m.key for m in MODELS], choices=list(MODELS_BY_KEY))
    args = ap.parse_args()
    download_benchmark()
    for key in args.models:
        spec = MODELS_BY_KEY[key]
        t0 = time.perf_counter()
        snapshot_download(spec.repo, allow_patterns=PATTERNS)
        print(f"{spec.label}: ok in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
