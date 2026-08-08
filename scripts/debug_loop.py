"""Is the 'ALL RIGHT' hallucination loop a batching artifact? Transcribe it alone (batch=1)."""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit import get_transcriber, load_pins
from asr_fairness_audit.data.edacc import load_edacc

# Find the offending utterance in the smoke set
rows_tx = [json.loads(l) for l in (ROOT / "results/whisper-small/transcripts.jsonl").open(encoding="utf-8")]
bad = next(r for r in rows_tx if r["ref_raw"].strip() == "ALL RIGHT")
idx = int(bad["utt_id"].split("-")[-1])
print(f"Offender: {bad['utt_id']} ref={bad['ref_raw']!r}")
print(f"Batched hyp was: {bad['hyp_raw'][:80]!r}...")

cs = load_edacc("test", load_pins())
row = cs.rows[idx]
assert row["text"].strip() == "ALL RIGHT", "index mismatch"

t = get_transcriber("whisper-small", load_pins())
t.batch_size = 1
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "one.wav"
    p.write_bytes(row["audio"]["bytes"])
    res = t.transcribe([str(p)])
print(f"Batch=1 hyp:     {res[0].text[:200]!r}")
