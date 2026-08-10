"""Can Moonshine Voice be made position-independent? Test isolation strategies.

For each strategy: transcribe a target utterance COLD (fresh state), run 20 other
utterances through, then transcribe the target again WARM. If cold == warm for
every target, that strategy isolates state and is usable for the audit.

    python scripts/test_moonshine_isolation.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

import moonshine_voice as mv  # noqa: E402

from asr_fairness_audit import load_pins  # noqa: E402
from asr_fairness_audit.backends.hf import TARGET_SR, load_16k_mono  # noqa: E402
from asr_fairness_audit.backends.moonshine import MoonshineVoiceTranscriber  # noqa: E402
from asr_fairness_audit.data.edacc import load_edacc  # noqa: E402

TARGETS = [12, 20, 31, 44]
WARMUP = list(range(0, 40, 2))


def audio_for(rows, i, tmp: Path) -> list:
    p = tmp / f"{i}.wav"
    p.write_bytes(rows[i]["audio"]["bytes"])
    a = load_16k_mono(str(p))
    p.unlink(missing_ok=True)
    return np.asarray(a, dtype=np.float32).tolist()


def text_of(transcript) -> str:
    lines = getattr(transcript, "lines", None) or []
    return " ".join((getattr(l, "text", "") or "").strip() for l in lines)


def main():
    import tempfile

    pins = load_pins()
    rows = load_edacc("test", pins).rows
    base = MoonshineVoiceTranscriber("UsefulSensors/moonshine-streaming-medium",
                                     revision=pins["models"]["UsefulSensors/moonshine-streaming-medium"])
    model_path, arch = base.model_path, base.arch

    def new_transcriber():
        return mv.Transcriber(model_path=model_path, model_arch=arch)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cache = {i: audio_for(rows, i, tmp) for i in set(TARGETS + WARMUP)}

    def run(t, audio, strategy):
        if strategy == "start_stop":
            t.start()
            out = t.transcribe_without_streaming(audio, TARGET_SR)
            t.stop()
            return text_of(out)
        return text_of(t.transcribe_without_streaming(audio, TARGET_SR))

    strategies = ["shared", "start_stop", "fresh_stream", "fresh_transcriber"]
    for strat in strategies:
        try:
            if strat == "fresh_transcriber":
                cold = {i: run(new_transcriber(), cache[i], strat) for i in TARGETS}
                for i in WARMUP:
                    run(new_transcriber(), cache[i], strat)
                warm = {i: run(new_transcriber(), cache[i], strat) for i in TARGETS}
            elif strat == "fresh_stream":
                t = new_transcriber()
                if not hasattr(t, "create_stream"):
                    print(f"{strat}: unsupported"); continue
                def via_stream(a):
                    s = t.create_stream()
                    s.start(); s.add_audio(a, TARGET_SR); s.stop()
                    tr = t.update_transcription()
                    return text_of(tr)
                cold = {i: via_stream(cache[i]) for i in TARGETS}
                for i in WARMUP:
                    via_stream(cache[i])
                warm = {i: via_stream(cache[i]) for i in TARGETS}
            else:
                t = new_transcriber()
                cold = {i: run(t, cache[i], strat) for i in TARGETS}
                for i in WARMUP:
                    run(t, cache[i], strat)
                warm = {i: run(t, cache[i], strat) for i in TARGETS}

            diffs = [i for i in TARGETS if cold[i] != warm[i]]
            print(f"{strat:18s}: {len(TARGETS) - len(diffs)}/{len(TARGETS)} stable"
                  f"{'' if not diffs else '  (differ: ' + str(diffs) + ')'}")
            if diffs:
                i = diffs[0]
                print(f"    cold: {cold[i][:80]!r}\n    warm: {warm[i][:80]!r}")
        except Exception as e:
            print(f"{strat:18s}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
