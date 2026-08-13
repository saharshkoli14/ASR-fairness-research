"""Check the prepared AfriSpeech subset before any training run (EVAL_SPEC §6).

    python scripts/verify_prepared.py --data-dir C:\\asr-data\\afrispeech-200

Cheap invariants, expensive to discover later. Each is a claim the fine-tuning
result would otherwise rest on silently:

  1. train / val speaker-disjoint  — a speaker in both makes validation worst-group
     WER optimistic, and that number selects the DRO step size (§6).
  2. train / test speaker-disjoint — AfriSpeech's published splits are assumed
     speaker-disjoint but this is not documented; overlap would inflate every
     fine-tuned number reported on the test split.
  3. every group has >= 20 min validation — the 2026-08-11 amendment.
  4. group imbalance preserved      — Group-DRO is being tested on it.
  5. audio on disk matches the manifest (sample rate, duration, non-silence).
  6. no empty transcripts.

Exit code is non-zero if any hard invariant fails.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parents[1]
SEED = 3407
MIN_VAL_MINUTES = 20.0
TARGET_SR = 16_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--audio-sample", type=int, default=60, help="files to open and check")
    args = ap.parse_args()

    prep = Path(args.data_dir) / "prepared"
    meta = json.loads((prep / "MANIFEST.json").read_text())
    rows = [json.loads(x) for x in (prep / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    groups = json.loads((ROOT / "groups_afrispeech.json").read_text())["groups"]

    fails, warns = [], []
    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)

    print(f"manifest: {len(rows)} utterances, seed {meta['seed']}, "
          f"sr {meta['sample_rate']}, revision {meta['dataset_revision'][:12]}")
    for s in ("train", "val", "test"):
        h = sum(r["duration"] for r in by_split[s]) / 3600
        spk = len({r["speaker"] for r in by_split[s]})
        print(f"  {s:5} {len(by_split[s]):6} utt  {h:7.2f} h  {spk:5} speakers")

    # 1 & 2 — speaker disjointness
    spk = {s: {r["speaker"] for r in by_split[s]} for s in by_split}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = spk.get(a, set()) & spk.get(b, set())
        if overlap:
            n_utt = sum(1 for r in by_split[b] if r["speaker"] in overlap)
            msg = (f"{a}/{b} share {len(overlap)} speaker(s), {n_utt} {b} utterances affected")
            (fails if b != "test" else warns).append(msg)
            print(f"  {'FAIL' if b != 'test' else 'WARN'}: {msg}")
        else:
            print(f"  ok: {a}/{b} speaker-disjoint")

    # 3 — validation minutes per group
    val_min = defaultdict(float)
    for r in by_split["val"]:
        val_min[r["accent"]] += r["duration"] / 60
    for g in groups:
        if val_min[g] < MIN_VAL_MINUTES:
            fails.append(f"val {g}: {val_min[g]:.1f} min < {MIN_VAL_MINUTES:.0f}")
            print(f"  FAIL: val {g} {val_min[g]:.1f} min")
    if not any(f.startswith("val ") for f in fails):
        print(f"  ok: every group >= {MIN_VAL_MINUTES:.0f} min validation "
              f"(min {min(val_min.values()):.1f})")

    # 4 — imbalance preserved
    train_h = defaultdict(float)
    for r in by_split["train"]:
        train_h[r["accent"]] += r["duration"] / 3600
    if set(train_h) != set(groups):
        fails.append(f"train groups {sorted(train_h)} != frozen groups {sorted(groups)}")
    ratio = max(train_h.values()) / max(min(train_h.values()), 1e-9)
    src = max(g["train"]["minutes"] for g in groups.values()) / \
        min(g["train"]["minutes"] for g in groups.values())
    print(f"  {'ok' if abs(ratio - src) < 1.0 else 'WARN'}: imbalance {ratio:.1f}x "
          f"(source {src:.1f}x)")
    if abs(ratio - src) >= 1.0:
        warns.append(f"imbalance drifted: {ratio:.1f}x vs source {src:.1f}x")
    for g in sorted(train_h, key=lambda k: -train_h[k]):
        print(f"      {g:10} {train_h[g]:6.2f} h  {val_min[g]:6.1f} min val")

    # 6 — transcripts
    empty = [r for r in rows if not r["text"].strip()]
    if empty:
        fails.append(f"{len(empty)} utterances have empty transcripts")
        print(f"  FAIL: {len(empty)} empty transcripts")
    else:
        print("  ok: no empty transcripts")

    # 5 — audio spot check
    try:
        import soundfile as sf
        rng = random.Random(SEED)
        sample = rng.sample(rows, min(args.audio_sample, len(rows)))
        bad_sr, bad_dur, silent, missing = 0, 0, 0, 0
        for r in sample:
            p = prep / r["path"].replace("\\", "/")   # manifest may be Windows-written
            if not p.exists():
                missing += 1
                continue
            info = sf.info(p)
            if info.samplerate != TARGET_SR:
                bad_sr += 1
            if abs(info.duration - r["duration"]) > 0.05:
                bad_dur += 1
            data, _ = sf.read(p, dtype="float32")
            if not data.size or float(abs(data).max()) < 1e-6:
                silent += 1
        print(f"  audio spot check ({len(sample)} files): missing {missing}, "
              f"wrong sr {bad_sr}, duration mismatch {bad_dur}, silent {silent}")
        for label, n in (("missing", missing), ("wrong sample rate", bad_sr),
                         ("duration mismatch", bad_dur), ("silent", silent)):
            if n:
                fails.append(f"{n} sampled file(s): {label}")
    except ImportError:
        warns.append("soundfile unavailable — audio not checked")

    print()
    for w in warns:
        print(f"WARN: {w}")
    if fails:
        print("\nFAILED — do not train on this subset:")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("All invariants hold. Subset is safe to train on.")


if __name__ == "__main__":
    main()
