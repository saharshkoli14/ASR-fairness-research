"""Pairwise model comparison with paired bootstrap CIs (EVAL_SPEC §4.4).

    python scripts/compare_models.py                       # all completed models
    python scripts/compare_models.py --metric worst_group_wer

Writes results/comparisons.json. A difference counts as supported only if its
95% CI excludes zero.
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit.compare import paired_bootstrap  # noqa: E402
from asr_fairness_audit.metrics import Utterance  # noqa: E402
from asr_fairness_audit.normalize import normalize, normalize_reference  # noqa: E402


def load_utts(model: str, groups: set[str]) -> list[Utterance]:
    tx = ROOT / "results" / model / "transcripts.jsonl"
    out = []
    for line in tx.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["accent"] not in groups:
            continue
        out.append(Utterance(ref=normalize_reference(r["ref_raw"]), hyp=normalize(r["hyp_raw"]),
                             group=r["accent"], speaker=r["speaker"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="gap_max_minus_min",
                    choices=["gap_max_minus_min", "worst_group_wer", "macro_wer"])
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="family-wise alpha; use 0.05/n_comparisons for Bonferroni")
    ap.add_argument("--bonferroni", action="store_true",
                    help="divide alpha by the number of comparisons actually run")
    args = ap.parse_args()

    groups = set(json.loads((ROOT / "groups.json").read_text())["groups"])
    models = args.models or sorted(
        d.name for d in (ROOT / "results").iterdir()
        if (d / "summary.json").exists() and (d / "transcripts.jsonl").exists()
    )
    print(f"models: {models}\nmetric: {args.metric}\n")

    utts = {m: load_utts(m, groups) for m in models}
    pairs = list(combinations(models, 2))
    alpha = args.alpha / len(pairs) if args.bonferroni else args.alpha
    if args.bonferroni:
        print(f"Bonferroni: {len(pairs)} comparisons, per-test alpha = {alpha:.5f} "
              f"({100 * (1 - alpha):.2f}% CIs)\n")
    results = []
    for a, b in pairs:
        r = paired_bootstrap(utts[a], utts[b], metric=args.metric, alpha=alpha)
        r["model_a"], r["model_b"] = a, b
        results.append(r)
        verdict = "SUPPORTED" if r["significant"] else "not supported (CI includes 0)"
        print(f"{a} vs {b}")
        print(f"  {args.metric}: {r['point_a'] * 100:.1f} vs {r['point_b'] * 100:.1f}  "
              f"diff {r['difference'] * 100:+.1f} pts  "
              f"95% CI [{r['ci_low'] * 100:+.1f}, {r['ci_high'] * 100:+.1f}]  -> {verdict}\n")

    out = ROOT / "results" / f"comparisons_{args.metric}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
