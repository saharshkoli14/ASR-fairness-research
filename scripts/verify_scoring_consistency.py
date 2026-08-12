"""Re-score every model from committed transcripts, in ONE interpreter (EVAL_SPEC §4.3, §7).

    python scripts/verify_scoring_consistency.py
    python scripts/verify_scoring_consistency.py --tolerance 0.0005

RESULTS.md claims every number is reproducible from `results/<model>/transcripts.jsonl`.
This checks that claim, and one specific threat to it: the audit ran across two
environments (NeMo models under WSL2, HF models on native Windows), and per-run
provenance added 2026-08-11 shows those environments carry **different major
versions of jiwer** — 3.1.0 under WSL, 4.0.0 on Windows. Since `summary.json`
records no library versions, there is no way to tell after the fact which version
computed which row of the headline table. If the two disagree at all, the
per-group WERs are not mutually comparable and the disparity metrics built on
them are unsafe.

Scoring is deterministic and CPU-only, so this reproduces the scoring path exactly
(same normalizer, same group filter, same metrics) without touching a GPU or a model.
A mismatch is reported per group; it does not re-write summary.json.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit.metrics import Utterance, evaluate  # noqa: E402
from asr_fairness_audit.normalize import normalize, normalize_reference, vendor_info  # noqa: E402
from asr_fairness_audit.provenance import library_versions  # noqa: E402


def rescore(model_dir: Path, audited: set[str]) -> dict:
    """Replay the scoring half of run_audit.py, verbatim, from the committed file."""
    records = [json.loads(line) for line in
               (model_dir / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()]
    utts = []
    for r in records:
        if r["accent"] not in audited:
            continue                      # pooled groups: appendix only (EVAL_SPEC §3)
        utts.append(Utterance(ref=normalize_reference(r["ref_raw"]), hyp=normalize(r["hyp_raw"]),
                              group=r["accent"], speaker=r["speaker"]))
    return {"n": len(utts), "metrics": evaluate(utts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, default=1e-6,
                    help="max absolute difference in WER treated as a match")
    args = ap.parse_args()

    groups = json.loads((ROOT / "groups.json").read_text())
    audited = set(groups["groups"])
    libs = library_versions()
    print(f"re-scoring with jiwer=={libs.get('jiwer','?')} numpy=={libs.get('numpy','?')} "
          f"normalizer={vendor_info()['commit'][:12]}\n")

    worst = 0.0
    failures = []
    for model_dir in sorted((ROOT / "results").iterdir()):
        summary_file = model_dir / "summary.json"
        if not summary_file.is_dir() and not summary_file.exists():
            continue
        committed = json.loads(summary_file.read_text())
        got = rescore(model_dir, audited)

        deltas = {}
        if got["n"] != committed["n_scored_utterances"]:
            failures.append(f"{model_dir.name}: scored {got['n']} utterances, "
                            f"summary.json says {committed['n_scored_utterances']}")
        old, new = committed["metrics"], got["metrics"]
        for key, val in new.items():
            if key == "per_group":          # {group: {wer, n_utterances, n_speakers, ref_words}}
                for g, stats in val.items():
                    ref = old.get("per_group", {}).get(g)
                    if ref is None:
                        deltas[f"per_group.{g}"] = float("inf")
                        continue
                    for field in ("wer", "n_utterances", "n_speakers", "ref_words"):
                        d = abs(stats[field] - ref[field])
                        if d > args.tolerance:
                            deltas[f"{g}.{field}"] = d
            elif isinstance(val, (int, float)) and key in old:
                d = abs(val - old[key])
                if d > args.tolerance:
                    deltas[key] = d

        worst = max([worst, *deltas.values()]) if deltas else worst
        if deltas:
            failures.append(f"{model_dir.name}: {len(deltas)} metric(s) differ, "
                            f"max Δ {max(deltas.values()):.6f} ({max(deltas, key=deltas.get)})")
            print(f"FAIL  {model_dir.name:30} max Δ {max(deltas.values()):.6f}")
        else:
            print(f"ok    {model_dir.name:30} {got['n']} utterances, all metrics reproduce")

    print()
    if failures:
        print("MISMATCHES — committed numbers do not reproduce in this interpreter:")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print(f"All models reproduce from committed transcripts (max Δ {worst:.2e}).")
    print("Scoring is stable across the jiwer versions used by the audit environments.")


if __name__ == "__main__":
    main()
