"""Paired speaker-level comparison of two fine-tuning arms (EVAL_SPEC §4.4, §6).

    python scripts/compare_arms.py \
        --a results/ft/erm/transcripts_test_step4200.jsonl \
        --b results/ft/dro_tau0.5/transcripts_test_step4200.jsonl \
        --label-a "ERM (step4200)" --label-b "DRO tau=0.5 (step4200)"

Both arms transcribe the SAME utterances, so the comparison must be paired: resample
speakers with replacement and recompute *both* arms' metrics on that identical
resample, then take the difference. Comparing two independent CIs instead — as the
per-arm bootstrap invites — discards the pairing and is far less sensitive; two
overlapping intervals do not imply the difference is indistinguishable from zero.

Resampling is over SPEAKERS, not utterances: utterances within a speaker are
correlated, and treating them as independent would understate the interval.
Same 1,000 resamples and seed 3407 as Part 1, so the numbers sit on the same footing.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit.metrics import Utterance, evaluate  # noqa: E402
from asr_fairness_audit.normalize import normalize, normalize_reference  # noqa: E402

SEED = 3407
N_BOOT = 1000
METRICS = ("worst_group_wer", "gap_max_minus_min", "macro_wer", "micro_wer")


def load(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r["path"]] = Utterance(ref=normalize_reference(r["ref"]), hyp=normalize(r["hyp"]),
                                   group=r["accent"], speaker=r["speaker"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A, B = load(Path(args.a)), load(Path(args.b))
    keys = sorted(set(A) & set(B))
    if len(keys) != len(A) or len(keys) != len(B):
        print(f"note: {len(A)} vs {len(B)} utterances, comparing the {len(keys)} in common")

    by_spk = defaultdict(list)
    for k in keys:
        by_spk[A[k].speaker].append(k)
    speakers = sorted(by_spk)

    point_a, point_b = evaluate([A[k] for k in keys]), evaluate([B[k] for k in keys])
    rng = np.random.default_rng(SEED)
    deltas = {m: [] for m in METRICS}
    for _ in range(N_BOOT):
        drawn = rng.choice(len(speakers), size=len(speakers), replace=True)
        ks = [k for i in drawn for k in by_spk[speakers[i]]]
        ra, rb = evaluate([A[k] for k in ks]), evaluate([B[k] for k in ks])
        for m in METRICS:
            deltas[m].append(rb[m] - ra[m])

    print(f"\npaired speaker bootstrap, n_speakers={len(speakers)}, "
          f"n_utterances={len(keys)}, {N_BOOT} resamples, seed {SEED}")
    print(f"delta = ({args.label_b}) - ({args.label_a}); negative favours {args.label_b}\n")
    print(f"{'metric':20} {'A':>8} {'B':>8} {'delta':>8}   99.67% CI          verdict")
    res = {}
    for m in METRICS:
        d = np.array(deltas[m]) * 100
        lo, hi = np.percentile(d, [0.165, 99.835])     # Bonferroni-style, as Part 1
        sig = "significant" if (lo > 0) == (hi > 0) else "indistinguishable"
        print(f"{m:20} {100 * point_a[m]:8.2f} {100 * point_b[m]:8.2f} "
              f"{100 * (point_b[m] - point_a[m]):+8.2f}   [{lo:+6.2f}, {hi:+6.2f}]   {sig}")
        res[m] = {"a": point_a[m], "b": point_b[m], "delta": point_b[m] - point_a[m],
                  "ci_low": lo / 100, "ci_high": hi / 100, "significant": sig == "significant"}

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"a": args.label_a, "b": args.label_b, "a_file": args.a, "b_file": args.b,
             "n_speakers": len(speakers), "n_utterances": len(keys),
             "n_boot": N_BOOT, "seed": SEED, "paired": True, "metrics": res}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
