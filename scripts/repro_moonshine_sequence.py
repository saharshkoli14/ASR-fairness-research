"""Reproduce Moonshine's sequence-dependent crash and test fixes.

Replays utterances 560..655 (the run died inside 640..655) sequentially in ONE
process, under three modes:
  baseline    — exactly the audit configuration
  cache-reset — drop any cached generate state on the model between utterances
  eager       — eager attention instead of SDPA

    python scripts/repro_moonshine_sequence.py baseline
    python scripts/repro_moonshine_sequence.py cache-reset
    python scripts/repro_moonshine_sequence.py eager
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit import load_pins  # noqa: E402
from asr_fairness_audit.backends.hf import HFTranscriber  # noqa: E402
from asr_fairness_audit.data.edacc import load_edacc  # noqa: E402

START, END = 560, 656


def drop_generate_caches(model):
    """Remove cached generation state transformers may keep on the model between calls."""
    for attr in ("_cache", "cache", "past_key_values"):
        if hasattr(model, attr):
            try:
                setattr(model, attr, None)
            except Exception:
                pass


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    pins = load_pins()
    rows = load_edacc("test", pins).rows
    t = HFTranscriber("UsefulSensors/moonshine-streaming-medium",
                      revision=pins["models"]["UsefulSensors/moonshine-streaming-medium"],
                      sdpa=(mode != "eager"), dtype="float32",
                      pad_to_multiple=80, chunk_s=30.0)

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(START, END):
            p = Path(tmp) / "x.wav"
            p.write_bytes(rows[i]["audio"]["bytes"])
            try:
                t.transcribe([str(p)])
            except Exception as e:
                print(f"\nCRASH at test-{i} in mode={mode}: {type(e).__name__}")
                sys.exit(1)
            if mode == "cache-reset":
                drop_generate_caches(t._pipe.model)
            print(".", end="", flush=True)
    print(f"\nmode={mode}: survived {END - START} sequential utterances")


if __name__ == "__main__":
    main()
