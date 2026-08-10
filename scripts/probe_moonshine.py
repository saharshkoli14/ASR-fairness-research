"""Confirm Moonshine's failure length. Transcribes synthetic audio of increasing
duration until it breaks, then reports the safe ceiling. Run in a FRESH process
(CUDA asserts poison the context).

    python scripts/probe_moonshine.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]

DURATIONS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0]  # short-end sweep: EdAcc has sub-second utterances

WORKER = r"""
import sys
sys.path.insert(0, r"{src}")
import numpy as np
from asr_fairness_audit import load_pins
from asr_fairness_audit.backends.hf import HFTranscriber

dur = float(sys.argv[1])
use_sdpa = sys.argv[2] == "sdpa"
pins = load_pins()
t = HFTranscriber("UsefulSensors/moonshine-streaming-medium",
                  revision=pins["models"]["UsefulSensors/moonshine-streaming-medium"],
                  sdpa=use_sdpa, dtype="float32", pad_to_multiple=80, chunk_s=None)
sr = 16000
x = (0.1 * np.sin(2 * np.pi * 220 * np.linspace(0, dur, int(sr * dur)))).astype("float32")
import soundfile as sf, tempfile, os
p = tempfile.mktemp(suffix=".wav"); sf.write(p, x, sr)
out = t.transcribe([p]); os.unlink(p)
print("OK", dur)
"""


def main():
    worker = WORKER.replace("{src}", str(ROOT / "src"))
    for attn in ("sdpa", "eager"):
        print(f"\n--- attention: {attn} ---")
        passing = []
        for d in DURATIONS:
            r = subprocess.run([sys.executable, "-c", worker, str(d), attn],
                               capture_output=True, text=True, timeout=600)
            ok = "OK" in r.stdout
            print(f"{d:>5.2f}s: {'ok' if ok else 'FAIL'}")
            if ok:
                passing.append(d)
        print(f"{attn}: shortest passing duration = {min(passing) if passing else 'NONE'}s")


if __name__ == "__main__":
    main()
