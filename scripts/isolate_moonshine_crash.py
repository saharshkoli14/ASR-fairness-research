"""Find the exact EdAcc utterance that crashes Moonshine, one subprocess per utterance.

Reads results/moonshine-streaming-medium/transcripts.jsonl to find where the run
died, then tries the next 16 utterances individually and reports which fail,
with audio properties of each.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit import load_pins  # noqa: E402
from asr_fairness_audit.data.edacc import load_edacc  # noqa: E402

WORKER = r"""
import sys, io
sys.path.insert(0, r"{src}")
import numpy as np, soundfile as sf
from asr_fairness_audit import load_pins
from asr_fairness_audit.backends.hf import HFTranscriber
from asr_fairness_audit.data.edacc import load_edacc

idx = int(sys.argv[1])
pins = load_pins()
row = load_edacc("test", pins).rows[idx]
data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
print(f"props dur={len(data)/sr:.2f}s sr={sr} min={data.min():.3f} max={data.max():.3f} "
      f"nan={np.isnan(data).any()} inf={np.isinf(data).any()}", flush=True)
t = HFTranscriber("UsefulSensors/moonshine-streaming-medium",
                  revision=pins["models"]["UsefulSensors/moonshine-streaming-medium"],
                  sdpa=True, dtype="float32", pad_to_multiple=80, chunk_s=30.0)
import tempfile, os
p = tempfile.mktemp(suffix=".wav"); sf.write(p, data, sr)
out = t.transcribe([p]); os.unlink(p)
print("OK:", out[0].text[:60])
"""


def main():
    tx = ROOT / "results/moonshine-streaming-medium/transcripts.jsonl"
    done = {json.loads(l)["utt_id"] for l in tx.read_text(encoding="utf-8").splitlines()} if tx.exists() else set()
    n_rows = len(load_edacc("test", load_pins()).rows)
    todo = [i for i in range(n_rows) if f"test-{i}" not in done]
    suspects = todo[:16]
    print(f"{len(done)} done; testing utterances {suspects[0]}..{suspects[-1]} individually\n")

    worker = WORKER.replace("{src}", str(ROOT / "src"))
    for i in suspects:
        r = subprocess.run([sys.executable, "-c", worker, str(i)],
                           capture_output=True, text=True, timeout=900)
        props = next((l for l in r.stdout.splitlines() if l.startswith("props")), "props ?")
        ok = "OK:" in r.stdout
        print(f"test-{i}: {'ok  ' if ok else 'FAIL'}  {props[6:]}")
        if not ok:
            tail = (r.stderr or r.stdout).strip().splitlines()
            print("   last error line:", tail[-1][:150] if tail else "?")


if __name__ == "__main__":
    main()
