"""Position-independence check: does a model's output depend on process state?

Re-transcribes a sample of already-saved utterances in a fresh process and diffs
against what the audit run produced mid-stream. Any mismatch means outputs depend
on how many prior calls the process made — results are not reproducible.

    python scripts/verify_determinism.py whisper-large-v3-turbo
    python scripts/verify_determinism.py distil-large-v3.5
    python scripts/verify_determinism.py whisper-small

Exit code 0 = deterministic, 1 = position-dependent.
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit import get_transcriber, load_pins  # noqa: E402
from asr_fairness_audit.data.edacc import load_edacc  # noqa: E402

N_SAMPLE = 25


def main():
    model = sys.argv[1]
    pins = load_pins()
    tx = ROOT / "results" / model / "transcripts.jsonl"
    saved = {json.loads(l)["utt_id"]: json.loads(l)["hyp_raw"]
             for l in tx.read_text(encoding="utf-8").splitlines()}
    ids = sorted(saved, key=lambda u: int(u.split("-")[-1]))
    # Sample across the whole run: state accumulation grows with position.
    sample = ids[:: max(1, len(ids) // N_SAMPLE)][:N_SAMPLE]

    rows = load_edacc("test", pins).rows
    t = get_transcriber(model, pins)

    mismatch = []
    with tempfile.TemporaryDirectory() as tmp:
        for u in sample:
            i = int(u.split("-")[-1])
            p = Path(tmp) / "x.wav"
            p.write_bytes(rows[i]["audio"]["bytes"])
            fresh = t.transcribe([str(p)])[0].text
            if fresh != saved[u]:
                mismatch.append((u, saved[u], fresh))

    for u, s, f in mismatch:
        print(f"DIFF {u}\n  saved: {s[:100]!r}\n  fresh: {f[:100]!r}")
    print(f"\n{model}: {len(sample) - len(mismatch)}/{len(sample)} identical")
    if mismatch:
        print("POSITION-DEPENDENT — results not reproducible")
        sys.exit(1)
    print("deterministic — results reproducible")


if __name__ == "__main__":
    main()
