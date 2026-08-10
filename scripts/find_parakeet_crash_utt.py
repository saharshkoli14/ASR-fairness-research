"""Identify the utterance that reliably crashes Parakeet, and test the duration hypothesis.

Both crashes stopped after 3264 saved rows, on the 13th item of the next batch,
pointing at index ~3277. Conformer rel_shift memory scales with sequence length^2,
so a very long utterance is the prime suspect.

    python scripts/find_parakeet_crash_utt.py
"""

import io
import json
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit import load_pins  # noqa: E402
from asr_fairness_audit.data.edacc import load_edacc  # noqa: E402

MODEL = "parakeet-tdt-0.6b-v2"

tx = ROOT / "results" / MODEL / "transcripts.jsonl"
done = sum(1 for _ in tx.open(encoding="utf-8")) if tx.exists() else 0
print(f"saved rows: {done}")

rows = load_edacc("test", load_pins()).rows
durs = []
for i in range(done, min(done + 16, len(rows))):
    d = sf.info(io.BytesIO(rows[i]["audio"]["bytes"])).duration
    durs.append((i, d, rows[i]["accent"], rows[i]["text"][:40]))

print(f"\nnext batch (indices {done}..{done + 15}):")
for i, d, acc, txt in durs:
    flag = "  <-- LONG" if d > 60 else ("  <- long" if d > 30 else "")
    print(f"  {i}: {d:7.2f}s  {acc[:22]:22s} {txt!r}{flag}")

longest = max(durs, key=lambda t: t[1])
print(f"\nlongest in batch: index {longest[0]} at {longest[1]:.1f}s")

print("\nscanning full split for long utterances (takes a minute)...")
alldur = [sf.info(io.BytesIO(r["audio"]["bytes"])).duration for r in rows]
for thr in (30, 60, 90, 120):
    n = sum(1 for d in alldur if d > thr)
    print(f"  > {thr:3d}s: {n:4d} utterances")
mx = max(alldur)
print(f"  max duration: {mx:.1f}s at index {alldur.index(mx)}")

(ROOT / "results" / MODEL / "durations.json").write_text(
    json.dumps({"batch_start": done, "batch": [(i, d) for i, d, _, _ in durs],
                "max_duration_s": mx}, indent=2))
