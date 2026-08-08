"""A/B: do whisper-small's hallucination loops persist at fp16 (the model-card default)?

Takes the first 60 loop-flagged utterances from the bf16 run and re-transcribes
them at fp16. If loops mostly vanish, bf16 was inflating them and all
Whisper-family runs must be redone at fp16. If they persist, they're real model
behavior and the bf16/fp16 choice was immaterial (still switching to fp16 for
spec compliance).
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit import get_transcriber, load_pins
from asr_fairness_audit.data.edacc import load_edacc
from asr_fairness_audit.normalize import normalize, normalize_reference

N = 60

summary = json.loads((ROOT / "results/whisper-small/summary.json").read_text(encoding="utf-8"))
loop_ids = [u for v in summary["hallucination_loops_by_group"].values() for u in v["utt_ids"]][:N]
idx = [int(u.split("-")[-1]) for u in loop_ids]

cs = load_edacc("test", load_pins())
t = get_transcriber("whisper-small", load_pins())  # registry now says fp16

still, fixed = 0, 0
with tempfile.TemporaryDirectory() as tmp:
    for i in idx:
        row = cs.rows[i]
        p = Path(tmp) / "x.wav"
        p.write_bytes(row["audio"]["bytes"])
        hyp = normalize(t.transcribe([str(p)])[0].text)
        ref = normalize_reference(row["text"])
        if len(hyp.split()) > max(10, 5 * len(ref.split())):
            still += 1
        else:
            fixed += 1

print(f"\n{len(idx)} bf16-loop utterances re-run at fp16: {still} still loop, {fixed} fixed")
print("VERDICT:", "loops were largely bf16-induced — RERUN whisper family at fp16" if fixed > still
      else "loops persist at fp16 — real model behavior; rerun still required for spec-compliant dtype")
