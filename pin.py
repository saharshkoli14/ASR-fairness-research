"""Freeze model checkpoint SHAs and dataset revisions into pins.json.

    python pin.py           # fails if pins.json exists (pins are frozen)
    python pin.py --force   # re-pin (requires a dated EVAL_SPEC.md changelog entry)

Every eval/train run must load with revision=pins["models"][repo_id] (or datasets
equivalent). Numbers produced without a pinned revision don't go in the paper.
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

PINS_FILE = Path(__file__).parent / "pins.json"

MODEL_REPOS = [
    "nvidia/canary-qwen-2.5b",
    "nvidia/parakeet-tdt-0.6b-v2",
    "nvidia/parakeet-tdt-0.6b-v3",
    "openai/whisper-large-v3-turbo",
    "distil-whisper/distil-large-v3.5",
    "UsefulSensors/moonshine-streaming-medium",
    "openai/whisper-small",
]

DATASET_REPOS = [
    "edinburghcstr/edacc",
    "intronhealth/afrispeech-200",
]


def env_snapshot() -> dict:
    snap = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "datasets", "nemo"):
        try:
            m = __import__(mod)
            snap[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            snap[mod] = None
    try:
        import torch

        if torch.cuda.is_available():
            snap["gpu"] = torch.cuda.get_device_properties(0).name
            snap["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if PINS_FILE.exists() and not args.force:
        sys.exit("pins.json exists — pins are frozen. Use --force only with a spec changelog entry.")

    api = HfApi()
    pins = {
        "pinned_at": datetime.now(timezone.utc).isoformat(),
        "models": {},
        "datasets": {},
        "env_at_pin_time": env_snapshot(),
    }
    for repo in MODEL_REPOS:
        sha = api.model_info(repo).sha
        pins["models"][repo] = sha
        print(f"model    {repo:45s} {sha}")
    for repo in DATASET_REPOS:
        sha = api.dataset_info(repo).sha
        pins["datasets"][repo] = sha
        print(f"dataset  {repo:45s} {sha}")

    PINS_FILE.write_text(json.dumps(pins, indent=2))
    print(f"\nWrote {PINS_FILE}")


if __name__ == "__main__":
    main()
