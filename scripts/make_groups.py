"""Freeze accent groups from the pinned EdAcc test split (EVAL_SPEC §3).

Inclusion rule: >= 20 minutes of test audio AND >= 3 distinct speakers.
Below-threshold groups pool into "other" (appendix only, never in disparity metrics).
Output groups.json is committed and is INPUT to all evals — never recomputed.

    python scripts/make_groups.py

Downloads EdAcc test audio (several GB) on first run; run on your machine.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from asr_fairness_audit import load_pins  # noqa: E402
from asr_fairness_audit.data.edacc import load_edacc  # noqa: E402

MIN_MINUTES = 20.0
MIN_SPEAKERS = 3
OUT = Path(__file__).parents[1] / "groups.json"


def main():
    if OUT.exists() and "--force" not in sys.argv:
        sys.exit("groups.json exists — groups are frozen. Use --force only with a spec changelog entry.")

    pins = load_pins()
    cs = load_edacc("test", pins)

    seconds: dict = defaultdict(float)
    speakers: dict = defaultdict(set)
    for row in cs.rows:
        audio = row["audio"]
        seconds[row["accent"]] += len(audio["array"]) / audio["sampling_rate"]
        speakers[row["accent"]].add(row["speaker"])

    groups, pooled = {}, {}
    for g in sorted(seconds):
        rec = {
            "minutes": round(seconds[g] / 60, 2),
            "n_speakers": len(speakers[g]),
            "n_utterances": cs.kept_by_group.get(g, 0),
        }
        if seconds[g] / 60 >= MIN_MINUTES and len(speakers[g]) >= MIN_SPEAKERS:
            groups[g] = rec
        else:
            pooled[g] = rec

    OUT.write_text(json.dumps({
        "dataset_revision": pins["datasets"]["edinburghcstr/edacc"],
        "split": "test",
        "rule": {"min_minutes": MIN_MINUTES, "min_speakers": MIN_SPEAKERS},
        "groups": groups,
        "pooled_into_other": pooled,
    }, indent=2))

    print(f"{len(groups)} groups pass, {len(pooled)} pooled into 'other'")
    for g, r in sorted(groups.items(), key=lambda kv: -kv[1]["minutes"]):
        print(f"  {g:40s} {r['minutes']:7.1f} min  {r['n_speakers']:3d} speakers")
    print(f"\nWrote {OUT} — commit it.")


if __name__ == "__main__":
    main()
