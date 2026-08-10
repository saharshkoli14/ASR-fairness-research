"""Does process position change Moonshine outputs, or only cause crashes?

Samples 20 already-saved transcripts (produced mid-run, with accumulated state),
re-transcribes them in this fresh process, and diffs. Identical outputs mean
restart-on-crash + resume is sound and saved rows are keepable.
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit import get_transcriber, load_pins  # noqa: E402
from asr_fairness_audit.data.edacc import load_edacc  # noqa: E402

pins = load_pins()
tx = ROOT / "results/moonshine-streaming-medium/transcripts.jsonl"
saved = {json.loads(l)["utt_id"]: json.loads(l)["hyp_raw"]
         for l in tx.read_text(encoding="utf-8").splitlines()}
ids = sorted(saved, key=lambda u: int(u.split("-")[-1]))
sample = ids[:: max(1, len(ids) // 20)][:20]

rows = load_edacc("test", pins).rows
t = get_transcriber("moonshine-streaming-medium", pins)

mismatch = 0
with tempfile.TemporaryDirectory() as tmp:
    for u in sample:
        i = int(u.split("-")[-1])
        p = Path(tmp) / "x.wav"
        p.write_bytes(rows[i]["audio"]["bytes"])
        fresh = t.transcribe([str(p)])[0].text
        if fresh != saved[u]:
            mismatch += 1
            print(f"DIFF {u}:\n  saved: {saved[u][:90]!r}\n  fresh: {fresh[:90]!r}")

print(f"\n{len(sample) - mismatch}/{len(sample)} identical.")
print("VERDICT:", "outputs are position-independent — restart+resume is sound, saved rows keepable"
      if mismatch == 0 else
      "outputs DEPEND on process state — saved rows are contaminated; wipe and rerun with process recycling")
