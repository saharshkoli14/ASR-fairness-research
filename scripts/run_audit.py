"""Accuracy audit runner (EVAL_SPEC §4.1, §5): one model over the cleaned EdAcc test split.

    python scripts/run_audit.py --model whisper-large-v3-turbo
    python scripts/run_audit.py --model whisper-large-v3-turbo --limit 50   # smoke run

Produces results/<model>/:
    transcripts.jsonl   one row per utterance (raw + normalized ref/hyp) — resumable checkpoint
    summary.json        per-group WER, micro/macro, worst-group, gap, std, bootstrap CIs
Run HF models on Windows, NeMo models under WSL2. Efficiency benchmarking is a
SEPARATE runner (EVAL_SPEC §4.2) — this one may batch however is convenient.
"""

import argparse
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit import MODELS, get_transcriber, load_pins  # noqa: E402
from asr_fairness_audit.data.edacc import exclusion_report, load_edacc  # noqa: E402
from asr_fairness_audit.metrics import Utterance, bootstrap_ci, evaluate  # noqa: E402
from asr_fairness_audit.normalize import normalize, normalize_reference, vendor_info  # noqa: E402

BATCH = 16  # utterances fetched per transcribe() call (backend batches internally)


def harness_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS.keys())
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="smoke-run on first N utterances")
    args = ap.parse_args()

    pins = load_pins()
    groups_file = ROOT / "groups.json"
    groups = json.loads(groups_file.read_text()) if groups_file.exists() else None
    if groups is None and args.limit is None:
        sys.exit("groups.json missing — run scripts/make_groups.py first (smoke runs with --limit are allowed).")

    out_dir = ROOT / "results" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    tx_file = out_dir / "transcripts.jsonl"

    # Resume: skip already-transcribed utterance ids.
    done: dict[str, dict] = {}
    if tx_file.exists():
        for line in tx_file.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["utt_id"]] = rec
        print(f"Resuming: {len(done)} utterances already transcribed.")

    print(f"Loading EdAcc {args.split} (pinned {pins['datasets']['edinburghcstr/edacc'][:12]})...")
    cs = load_edacc(args.split, pins)
    rows = cs.rows[: args.limit] if args.limit else cs.rows
    todo = [(i, r) for i, r in enumerate(rows) if f"{args.split}-{i}" not in done]
    print(f"{len(rows)} clean utterances, {len(todo)} to transcribe.")

    if todo:
        transcriber = get_transcriber(args.model, pins)
        t0 = time.time()
        with tx_file.open("a", encoding="utf-8") as fh, tempfile.TemporaryDirectory() as tmp:
            for start in range(0, len(todo), BATCH):
                chunk = todo[start:start + BATCH]
                paths = []
                for i, row in chunk:
                    p = Path(tmp) / f"{i}.wav"
                    p.write_bytes(row["audio"]["bytes"])
                    paths.append(str(p))
                results = transcriber.transcribe(paths)
                for (i, row), res in zip(chunk, results):
                    rec = {
                        "utt_id": f"{args.split}-{i}",
                        "speaker": row["speaker"],
                        "accent": row["accent"],
                        "ref_raw": row["text"],
                        "hyp_raw": res.text,
                        "detected_language": res.detected_language,
                        "meta": res.meta or None,
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                for p in paths:
                    Path(p).unlink(missing_ok=True)
                done_n = min(start + BATCH, len(todo))
                rate = done_n / (time.time() - t0)
                print(f"\r{done_n}/{len(todo)}  ({rate:.1f} utt/s, ~{(len(todo) - done_n) / max(rate, 0.01) / 60:.0f} min left)",
                      end="", flush=True)
        print()
        del transcriber

    # Score from the full transcript file (normalization happens here, never stored raw-free).
    records = [json.loads(line) for line in tx_file.read_text(encoding="utf-8").splitlines()]
    # Smoke runs (--limit) score every accent they see; real runs use frozen groups only.
    if groups and not args.limit:
        audited_groups = set(groups["groups"])
    else:
        audited_groups = {r["accent"] for r in records}
    utts, lang_misdetect = [], 0
    loops: dict[str, list] = {}
    chunked: dict[str, int] = {}
    for r in records:
        group = r["accent"] if r["accent"] in audited_groups else "other"
        if group == "other":
            continue  # pooled groups: appendix only, never in disparity metrics (EVAL_SPEC §3)
        if r.get("detected_language") and "en" not in r["detected_language"].lower():
            lang_misdetect += 1
        if (r.get("meta") or {}).get("chunked"):
            chunked[group] = chunked.get(group, 0) + 1
        ref_n, hyp_n = normalize_reference(r["ref_raw"]), normalize(r["hyp_raw"])
        # Hallucination-loop diagnostic (secondary; WER keeps these — deployed-default behavior):
        # hypothesis blows past 5x the reference length (min 10 words to skip trivial cases).
        if len(hyp_n.split()) > max(10, 5 * len(ref_n.split())):
            loops.setdefault(group, []).append(r["utt_id"])
        utts.append(Utterance(ref=ref_n, hyp=hyp_n, group=group, speaker=r["speaker"]))

    summary = {
        "model": args.model,
        "repo_id": MODELS[args.model][0],
        "model_revision": pins["models"][MODELS[args.model][0]],
        "dataset_revision": pins["datasets"]["edinburghcstr/edacc"],
        "split": args.split,
        "limit": args.limit,
        "harness_commit": harness_commit(),
        "normalizer_source": vendor_info()["commit"],
        "n_scored_utterances": len(utts),
        "language_misdetections": lang_misdetect,
        "hallucination_loops_by_group": {g: {"count": len(v), "utt_ids": v} for g, v in loops.items()},
        # Long-audio chunking is non-random w.r.t. accent (long turns cluster by speaker),
        # so it is reported per group like exclusions (EVAL_SPEC §5).
        "chunked_by_group": chunked,
        "metrics": evaluate(utts),
        "bootstrap": {m: bootstrap_ci(utts, metric=m)
                      for m in ("gap_max_minus_min", "worst_group_wer", "macro_wer")},
        "exclusions": exclusion_report(cs),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    m = summary["metrics"]
    print(f"\n=== {args.model} on EdAcc {args.split}" + (f" (LIMIT {args.limit} — smoke run, not reportable)" if args.limit else "") + " ===")
    print(f"micro WER {m['micro_wer']:.3f} | macro WER {m['macro_wer']:.3f} | "
          f"worst {m['worst_group_wer']:.3f} ({m['worst_group']}) | gap {m['gap_max_minus_min']:.3f}")
    print(f"Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
