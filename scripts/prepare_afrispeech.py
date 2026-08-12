"""Build the training-ready AfriSpeech subset (EVAL_SPEC §6, amendments 2026-08-11).

    python scripts/prepare_afrispeech.py --data-dir C:\\asr-data\\afrispeech-200 --inspect
    python scripts/prepare_afrispeech.py --data-dir ... --plan      # counts only, no audio
    python scripts/prepare_afrispeech.py --data-dir ...             # extract + resample + cache

Three stages, all seeded (3407) and all recorded in prepared/MANIFEST.json:

  1. **Validation carve.** Any audited group with < 20 min of dev audio takes a
     speaker-disjoint holdout from train until it reaches 20 min. Only hausa (15.4 min)
     qualifies. Held-out speakers are removed from training entirely — a speaker
     appearing in both would make validation worst-group WER optimistic, and that
     number selects the DRO step size.
  2. **Proportional subsample** of the remaining train to ~25 h, per group, stratified
     by speaker so the subsample keeps a group's speaker diversity rather than a few
     talkative individuals. The 11.7x group imbalance is preserved on purpose.
  3. **Extract + resample** the selected utterances only, 44.1 kHz -> 16 kHz mono,
     written as flac (~2.5x smaller than wav, lossless).

`--plan` stops after stage 2, so the selection can be checked before spending an hour
of I/O on 32 GB of tarballs.
"""

import argparse
import io
import json
import random
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEED = 3407
TARGET_TRAIN_HOURS = 25.0
MIN_VAL_MINUTES = 20.0
TARGET_SR = 16_000


def load_rows(data_dir: Path, split: str) -> list[dict]:
    import csv
    with (data_dir / "transcripts" / f"{split}.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def col(rows: list[dict], *names: str) -> str:
    have = {c.lower(): c for c in rows[0]}
    for n in names:
        if n in have:
            return have[n]
    sys.exit(f"none of {names} in columns: {', '.join(rows[0])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=None, help="default: <data-dir>/prepared")
    ap.add_argument("--hours", type=float, default=TARGET_TRAIN_HOURS)
    ap.add_argument("--plan", action="store_true", help="select utterances, write no audio")
    ap.add_argument("--inspect", action="store_true", help="show CSV columns and tar members")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out) if args.out else data_dir / "prepared"
    groups = json.loads((ROOT / "groups_afrispeech.json").read_text())
    audited = list(groups["groups"])

    rows = {s: load_rows(data_dir, s) for s in ("train", "dev", "test")}
    c_acc = col(rows["train"], "accent")
    c_spk = col(rows["train"], "user_ids", "speaker_id", "user_id")
    c_dur = col(rows["train"], "duration")
    c_txt = col(rows["train"], "transcript", "text")
    c_path = col(rows["train"], "audio_paths", "path", "audio_id")

    if args.inspect:
        print("columns:", ", ".join(rows["train"][0]))
        print(f"using accent={c_acc} speaker={c_spk} duration={c_dur} "
              f"text={c_txt} path={c_path}")
        print("\nsample path values:")
        for r in rows["train"][:3]:
            print("  ", r[c_path])
        tars = sorted((data_dir / "audio").rglob("*.tar.gz"))
        print(f"\n{len(tars)} tarballs; members of {tars[0].name}:" if tars else "no tarballs")
        if tars:
            with tarfile.open(tars[0]) as tf:
                for m in tf.getmembers()[:5]:
                    print("  ", m.name)
        return

    def norm(r):
        try:
            d = float(r[c_dur])
        except (TypeError, ValueError):
            return None
        acc = (r[c_acc] or "").strip().lower()
        if acc not in audited or d <= 0:
            return None
        return {"accent": acc, "speaker": r[c_spk], "duration": d,
                "text": (r[c_txt] or "").strip(), "src": r[c_path],
                "key": Path(str(r[c_path]).replace("\\", "/")).name}

    train = [x for x in map(norm, rows["train"]) if x]
    dev = [x for x in map(norm, rows["dev"]) if x]
    test = [x for x in map(norm, rows["test"]) if x]
    rng = random.Random(SEED)

    # --- stage 1: speaker-disjoint validation carve ------------------------------
    dev_min = defaultdict(float)
    for x in dev:
        dev_min[x["accent"]] += x["duration"] / 60
    by_spk = defaultdict(lambda: defaultdict(list))
    for x in train:
        by_spk[x["accent"]][x["speaker"]].append(x)

    carved, carved_speakers = [], defaultdict(set)
    for acc in audited:
        deficit = MIN_VAL_MINUTES - dev_min[acc]
        if deficit <= 0:
            continue
        speakers = sorted(by_spk[acc])
        rng.shuffle(speakers)
        got = 0.0
        for spk in speakers:
            if got >= deficit:
                break
            utts = by_spk[acc][spk]
            carved += utts
            carved_speakers[acc].add(spk)
            got += sum(u["duration"] for u in utts) / 60
        print(f"val carve {acc}: dev {dev_min[acc]:.1f} min + {got:.1f} min "
              f"from {len(carved_speakers[acc])} held-out train speaker(s)")

    train = [x for x in train if x["speaker"] not in carved_speakers[x["accent"]]]
    val = dev + carved

    # --- stage 2: proportional, speaker-stratified subsample ---------------------
    total_h = sum(x["duration"] for x in train) / 3600
    frac = min(1.0, args.hours / total_h)
    print(f"\ntrain pool {total_h:.1f} h -> target {args.hours:.1f} h (keep {frac:.1%} per group)")

    selected = []
    for acc in audited:
        pool = defaultdict(list)
        for x in train:
            if x["accent"] == acc:
                pool[x["speaker"]].append(x)
        want = sum(x["duration"] for g in pool.values() for x in g) * frac
        got, picked = 0.0, []
        speakers = sorted(pool)
        rng.shuffle(speakers)
        # round-robin across speakers: preserves speaker diversity at any subsample size
        cursors = {s: 0 for s in speakers}
        for s in speakers:
            rng.shuffle(pool[s])
        while got < want:
            progressed = False
            for s in speakers:
                if got >= want:
                    break
                i = cursors[s]
                if i < len(pool[s]):
                    picked.append(pool[s][i])
                    got += pool[s][i]["duration"]
                    cursors[s] = i + 1
                    progressed = True
            if not progressed:
                break
        selected += picked
        print(f"  {acc:10} {got / 3600:6.2f} h  {len(picked):6} utt  "
              f"{len({p['speaker'] for p in picked}):4} speakers")

    sel_h = sum(x["duration"] for x in selected) / 3600
    ratio = max(sum(x["duration"] for x in selected if x["accent"] == a) for a in audited) / \
        max(min(sum(x["duration"] for x in selected if x["accent"] == a) for a in audited), 1e-9)
    print(f"\nselected train {sel_h:.2f} h, imbalance {ratio:.1f}x "
          f"(source 11.7x), val {sum(x['duration'] for x in val) / 60:.1f} min, "
          f"test {sum(x['duration'] for x in test) / 3600:.2f} h")

    splits = {"train": selected, "val": val, "test": test}
    if args.plan:
        print("\n--plan: no audio written")
        return

    # --- stage 3: extract + resample --------------------------------------------
    import soundfile as sf
    import soxr

    # One sequential pass per tarball ("r|gz" streaming). Random access into a .tar.gz
    # re-decompresses from the start for every member, which turns 32 GB of source into
    # hours; streaming decompresses each archive exactly once.
    wanted: dict[str, tuple[str, dict]] = {}
    for split, items in splits.items():
        for x in items:
            wanted[x["key"]] = (split, x)
    print(f"\nneed {len(wanted)} utterances; streaming tarballs once each")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, found = [], set()
    tars = sorted((data_dir / "audio").rglob("*.tar.gz"))
    for n, tar in enumerate(tars, 1):
        hits = 0
        with tarfile.open(tar, "r|gz") as tf:      # stream: read each member in order
            for m in tf:
                if not m.isfile():
                    continue
                key = Path(m.name).name
                job = wanted.get(key)
                if job is None or key in found:
                    continue
                split, x = job
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                data, sr = sf.read(io.BytesIO(fh.read()), dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if sr != TARGET_SR:
                    data = soxr.resample(data, sr, TARGET_SR)
                dest = out_dir / split / x["accent"]
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / (Path(key).stem + ".flac")
                sf.write(path, data, TARGET_SR)
                manifest.append({"split": split, "accent": x["accent"], "speaker": x["speaker"],
                                 "duration": round(len(data) / TARGET_SR, 3),
                                 "text": x["text"], "path": str(path.relative_to(out_dir))})
                found.add(key)
                hits += 1
        print(f"  [{n:3}/{len(tars)}] {tar.name:44} {hits:5} kept  "
              f"({len(found)}/{len(wanted)} total)", flush=True)

    missing = len(wanted) - len(found)

    (out_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest), encoding="utf-8")
    (out_dir / "MANIFEST.json").write_text(json.dumps({
        "seed": SEED, "target_train_hours": args.hours, "min_val_minutes": MIN_VAL_MINUTES,
        "sample_rate": TARGET_SR, "groups": audited,
        "dataset_revision": groups["dataset_revision"],
        "carved_val_speakers": {k: sorted(v) for k, v in carved_speakers.items()},
        "n_utterances": {s: sum(1 for m in manifest if m["split"] == s) for s in splits},
        "hours": {s: round(sum(m["duration"] for m in manifest if m["split"] == s) / 3600, 3)
                  for s in splits},
        "missing_audio": missing,
    }, indent=2))
    print(f"\nwrote {out_dir}  (missing audio: {missing})")


if __name__ == "__main__":
    main()
