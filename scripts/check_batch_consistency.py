"""Verify batched inference matches batch=1 on the smoke set (EVAL_SPEC §5 sanity).

If outputs differ, batching changes model behavior and accuracy runs must use batch=1.
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit import get_transcriber, load_pins
from asr_fairness_audit.data.edacc import load_edacc

MODEL = "whisper-small"
N = 50

pins = load_pins()
batched = {json.loads(l)["utt_id"]: json.loads(l)["hyp_raw"]
           for l in (ROOT / f"results/{MODEL}/transcripts.jsonl").open(encoding="utf-8")}

cs = load_edacc("test", pins)
t = get_transcriber(MODEL, pins)
t.batch_size = 1

mismatches = 0
with tempfile.TemporaryDirectory() as tmp:
    for i, row in enumerate(cs.rows[:N]):
        p = Path(tmp) / f"{i}.wav"
        p.write_bytes(row["audio"]["bytes"])
        hyp1 = t.transcribe([str(p)])[0].text
        hyp8 = batched.get(f"test-{i}")
        if hyp1 != hyp8:
            mismatches += 1
            print(f"MISMATCH test-{i}:")
            print(f"  batch=8: {hyp8[:100]!r}")
            print(f"  batch=1: {hyp1[:100]!r}")
        p.unlink()

print(f"\n{N - mismatches}/{N} identical. " +
      ("Batching is output-safe on this sample." if mismatches == 0 else
       f"{mismatches} DIFFER — accuracy runs must use batch_size=1."))
