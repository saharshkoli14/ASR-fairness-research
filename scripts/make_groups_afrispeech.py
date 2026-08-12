"""Freeze AfriSpeech-200 accent groups for the fine-tuning experiment (EVAL_SPEC §3, §6).

    python scripts/make_groups_afrispeech.py --data-dir C:\\asr-data\\afrispeech-200
    python scripts/make_groups_afrispeech.py --data-dir ... --inspect   # show CSV columns only

Writes groups_afrispeech.json (committed; INPUT to training and eval, never recomputed)
and prints the exact `fetch_afrispeech.py --accents ...` command for the train audio,
so only qualifying accents are downloaded instead of all 50 GB.

Inclusion rule, applied per split: >= 20 min of audio AND >= 3 distinct speakers,
matching the EdAcc rule in §3. An accent must qualify in **train** (enough data to
be a Group-DRO group) and in **test** (enough data to report a per-group WER on).
Both are printed separately so the joint rule's cost is visible rather than assumed.

NOTE: applying §3's thresholds to a second corpus is a spec decision, not a
mechanical consequence of the frozen spec — changelog it before any training run.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit import load_pins  # noqa: E402

MIN_MINUTES = 20.0
MIN_SPEAKERS = 3
SPLITS = ("train", "dev", "test")

# The CSV schema is not documented in the dataset card; detect it rather than guess.
CANDIDATES = {
    "accent": ["accent", "accents", "native_language", "language"],
    "speaker": ["speaker_id", "user_ids", "user_id", "speaker", "client_id"],
    "duration": ["duration", "audio_duration", "seconds", "length"],
    "utt": ["audio_id", "audio_ids", "path", "audio_paths", "id"],
}


def pick(header: list[str], role: str) -> str | None:
    lower = {h.lower().strip(): h for h in header}
    for cand in CANDIDATES[role]:
        if cand in lower:
            return lower[cand]
    return None


def read_split(data_dir: Path, split: str, inspect: bool) -> list[dict]:
    path = data_dir / "transcripts" / f"{split}.csv"
    if not path.exists():
        sys.exit(f"missing {path} — run fetch_afrispeech.py --get transcripts first")
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{path} is empty")
    header = list(rows[0])
    if inspect:
        print(f"\n{split}.csv — {len(rows)} rows")
        print("  columns:", ", ".join(header))
        print("  first row:", {k: str(v)[:40] for k, v in list(rows[0].items())[:8]})
        return []
    cols = {role: pick(header, role) for role in CANDIDATES}
    missing = [r for r, c in cols.items() if c is None and r != "utt"]
    if missing:
        sys.exit(f"could not identify column(s) {missing} in {path}\n"
                 f"  available: {', '.join(header)}\n"
                 f"  add the real name to CANDIDATES in this script")
    out = []
    for r in rows:
        try:
            dur = float(r[cols["duration"]])
        except (TypeError, ValueError):
            continue
        out.append({"accent": (r[cols["accent"]] or "").strip().lower(),
                    "speaker": r[cols["speaker"]], "duration": dur})
    print(f"  {split}: {len(out)} utterances "
          f"(accent={cols['accent']}, speaker={cols['speaker']}, duration={cols['duration']})")
    return out


def tally(rows: list[dict]) -> dict:
    mins, spk, n = defaultdict(float), defaultdict(set), defaultdict(int)
    for r in rows:
        mins[r["accent"]] += r["duration"] / 60.0
        spk[r["accent"]].add(r["speaker"])
        n[r["accent"]] += 1
    return {a: {"minutes": round(mins[a], 2), "n_speakers": len(spk[a]), "n_utterances": n[a]}
            for a in mins}


def qualifies(stat: dict) -> bool:
    return stat["minutes"] >= MIN_MINUTES and stat["n_speakers"] >= MIN_SPEAKERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--inspect", action="store_true", help="print CSV columns and exit")
    ap.add_argument("--out", default=str(ROOT / "groups_afrispeech.json"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    stats = {}
    for split in SPLITS:
        rows = read_split(data_dir, split, args.inspect)
        if not args.inspect:
            stats[split] = tally(rows)
    if args.inspect:
        return

    train_ok = {a for a, s in stats["train"].items() if qualifies(s)}
    test_ok = {a for a, s in stats["test"].items() if qualifies(s)}
    keep = sorted(train_ok & test_ok)

    print(f"\ninclusion: >= {MIN_MINUTES:.0f} min AND >= {MIN_SPEAKERS} speakers, per split")
    print(f"  accents seen (train/test):        {len(stats['train'])} / {len(stats['test'])}")
    print(f"  qualify in train:                 {len(train_ok)}")
    print(f"  qualify in test:                  {len(test_ok)}")
    print(f"  qualify in BOTH (audited groups): {len(keep)}\n")

    hours = sum(stats["train"][a]["minutes"] for a in keep) / 60
    print(f"{'accent':28} {'train min':>10} {'spk':>5} {'test min':>10} {'spk':>5}")
    for a in keep:
        t, e = stats["train"][a], stats["test"][a]
        print(f"{a:28} {t['minutes']:10.1f} {t['n_speakers']:5} "
              f"{e['minutes']:10.1f} {e['n_speakers']:5}")
    print(f"\ntrain audio in audited groups: {hours:.1f} h "
          f"({100 * hours / max(sum(s['minutes'] for s in stats['train'].values()) / 60, 1e-9):.0f}% "
          f"of the train split)")

    doc = {
        "dataset": "intronhealth/afrispeech-200",
        "dataset_revision": load_pins()["datasets"]["intronhealth/afrispeech-200"],
        "rule": {"min_minutes": MIN_MINUTES, "min_speakers": MIN_SPEAKERS,
                 "applied_to": "train AND test independently"},
        "groups": {a: {"train": stats["train"][a], "test": stats["test"][a],
                       "dev": stats["dev"].get(a)} for a in keep},
        "pooled_into_other": sorted(set(stats["train"]) - set(keep)),
        "train_hours_audited": round(hours, 2),
    }
    Path(args.out).write_text(json.dumps(doc, indent=2))
    print(f"\nwrote {args.out}")
    print("\nfetch only the audio you need:\n"
          f"  python scripts/fetch_afrispeech.py --get train --accents {' '.join(keep)} "
          f"--dest {args.data_dir}")


if __name__ == "__main__":
    main()
