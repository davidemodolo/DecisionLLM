import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_flash_linear_attention_disabled_without_cuda():
    code = "import importlib, core\ntry:\n    importlib.import_module('fla')\n    print('fla')\nexcept ImportError:\n    print('torch')"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "torch"
