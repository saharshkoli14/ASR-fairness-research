"""Quantify a model's nondeterminism band (EVAL_SPEC §4.5 step 2).

Re-transcribes a fixed subset of a completed run a second time under identical
configuration and reports utterance-level disagreement and |dWER| overall and
per group. Writes results/<model>/nondeterminism.json.

    python scripts/measure_nondeterminism.py moonshine-streaming-medium --n 300
"""

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit import get_transcriber, load_pins  # noqa: E402
from asr_fairness_audit.data.edacc import load_edacc  # noqa: E402
from asr_fairness_audit.metrics import _counts  # noqa: E402
from asr_fairness_audit.normalize import normalize, normalize_reference  # noqa: E402


def wer(pairs):
    e = sum(_counts(r, h)[0] for r, h in pairs)
    w = sum(_counts(r, h)[1] for r, h in pairs)
    return e / w if w else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    pins = load_pins()
    out_dir = ROOT / "results" / args.model
    records = [json.loads(l) for l in (out_dir / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()]
    # Restrict to audited groups: pooled groups have n as low as 1, where a single
    # flipped utterance produces a meaningless 50-point "band" (EVAL_SPEC §3).
    audited = set(json.loads((ROOT / "groups.json").read_text())["groups"])
    records = [r for r in records if r["accent"] in audited]
    # Evenly spaced subset across the run, so the sample isn't clustered in one conversation.
    step = max(1, len(records) // args.n)
    subset = records[::step][: args.n]

    rows = load_edacc("test", pins).rows
    t = get_transcriber(args.model, pins)

    disagree, pass1, pass2 = 0, [], []
    by_group = defaultdict(lambda: {"pass1": [], "pass2": [], "disagree": 0, "n": 0})
    with tempfile.TemporaryDirectory() as td:
        for k, r in enumerate(subset, 1):
            i = int(r["utt_id"].split("-")[-1])
            p = Path(td) / "x.wav"
            p.write_bytes(rows[i]["audio"]["bytes"])
            hyp2 = t.transcribe([str(p)])[0].text
            ref = normalize_reference(r["ref_raw"])
            h1, h2 = normalize(r["hyp_raw"]), normalize(hyp2)
            g = r["accent"]
            if h1 != h2:
                disagree += 1
                by_group[g]["disagree"] += 1
            by_group[g]["n"] += 1
            by_group[g]["pass1"].append((ref, h1))
            by_group[g]["pass2"].append((ref, h2))
            pass1.append((ref, h1))
            pass2.append((ref, h2))
            if k % 25 == 0:
                print(f"\r{k}/{len(subset)}  disagreements: {disagree}", end="", flush=True)
    print()

    result = {
        "model": args.model,
        "n_utterances": len(subset),
        "disagreement_rate": round(disagree / len(subset), 4),
        "wer_pass1": round(wer(pass1), 4),
        "wer_pass2": round(wer(pass2), 4),
        "abs_delta_wer": round(abs(wer(pass1) - wer(pass2)), 4),
        "per_group": {
            g: {
                "n": v["n"],
                "disagreement_rate": round(v["disagree"] / v["n"], 4),
                "abs_delta_wer": round(abs(wer(v["pass1"]) - wer(v["pass2"])), 4),
            }
            for g, v in sorted(by_group.items())
        },
    }
    (out_dir / "nondeterminism.json").write_text(json.dumps(result, indent=2))

    print(f"\n=== {args.model} nondeterminism band (n={len(subset)}) ===")
    print(f"disagreement rate: {result['disagreement_rate']:.1%}")
    print(f"WER pass1 {result['wer_pass1']:.4f} | pass2 {result['wer_pass2']:.4f} | "
          f"|dWER| {result['abs_delta_wer']:.4f} ({result['abs_delta_wer'] * 100:.2f} points)")
    for g, v in result["per_group"].items():
        print(f"  {g:25s} n={v['n']:4d}  disagree {v['disagreement_rate']:5.1%}  "
              f"|dWER| {v['abs_delta_wer'] * 100:.2f} pts")


if __name__ == "__main__":
    main()
