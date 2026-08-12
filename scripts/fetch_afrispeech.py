"""Size, then fetch, AfriSpeech-200 at the pinned revision (EVAL_SPEC §6).

    python scripts/fetch_afrispeech.py                    # sizes only, no download
    python scripts/fetch_afrispeech.py --get transcripts  # ~MBs, unblocks group freezing
    python scripts/fetch_afrispeech.py --get dev test     # eval audio
    python scripts/fetch_afrispeech.py --get train        # the big one

Sizes come from the Hub API without downloading anything, because the dataset card
is not trustworthy here: it declares download_size ~1.4 GB, which is impossible for
200 h of 44.1 kHz audio (~60 GB uncompressed), while the repo reports 467 GB of
total storage across revisions. Check the real number before spending the disk.

Downloads are resumable and pinned to pins.json, so an interrupted run continues
rather than restarting, and the bytes are the same ones the audit will cite.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit import load_pins  # noqa: E402

REPO = "intronhealth/afrispeech-200"
DEST = ROOT / "data" / "afrispeech-200"      # gitignored; large


# huggingface_hub stages downloads at
#   <dest>\.cache\huggingface\download\<dir>\<28-char hash>.<64 hex>.<8 hex>.incomplete
# The staging filename is a fixed ~113 chars regardless of the original name, so a
# deep destination blows Windows' 260-char MAX_PATH partway through a multi-GB fetch.
STAGING_OVERHEAD = len(r".cache\huggingface\download") + 113 + 2
WINDOWS_MAX_PATH = 260


def check_path_budget(dest: str, filenames: list[str]) -> None:
    """Fail fast on Windows if the destination is too deep to stage into."""
    if os.name != "nt":
        return
    deepest = max(filenames, key=lambda f: len(os.path.dirname(f)))
    needed = len(str(Path(dest).resolve())) + 1 + len(os.path.dirname(deepest)) + STAGING_OVERHEAD
    if needed <= WINDOWS_MAX_PATH:
        return
    sys.exit(
        f"Destination too deep for Windows MAX_PATH: staging needs ~{needed} chars "
        f"(limit {WINDOWS_MAX_PATH}).\n"
        f"  dest: {dest}\n"
        f"  deepest entry: {os.path.dirname(deepest)}\n\n"
        f"Fix either way:\n"
        f"  * short destination (no admin):  --dest C:\\asr-data\\afrispeech-200\n"
        f"    58 GB does not belong inside the repo tree anyway.\n"
        f"  * or enable long paths (admin, permanent): set LongPathsEnabled=1 under\n"
        f"    HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem, then reopen the shell.")


def split_of(fname: str) -> str:
    if fname.startswith("transcripts/"):
        return "transcripts"
    for s in ("train", "dev", "test"):
        if f"/{s}/" in fname:
            return s
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--get", nargs="*", default=[], choices=["transcripts", "train", "dev", "test"],
                    help="splits to download; omit to only report sizes")
    ap.add_argument("--accents", nargs="*", default=None, metavar="ACCENT",
                    help="restrict audio to these accent directories (e.g. yoruba igbo hausa). "
                         "Most of AfriSpeech's ~120 accents fail the EVAL_SPEC §3 inclusion rule "
                         "and pool into 'other', so their tarballs are dead weight.")
    ap.add_argument("--dest", default=str(DEST))
    args = ap.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    revision = load_pins()["datasets"][REPO]
    info = HfApi().dataset_info(REPO, revision=revision, files_metadata=True)

    by_split: dict[str, list[int]] = {}
    for f in info.siblings:
        by_split.setdefault(split_of(f.rfilename), []).append(f.size or 0)

    print(f"{REPO} @ {revision[:12]}\n")
    total = 0
    for split in ("transcripts", "train", "dev", "test", "other"):
        sizes = by_split.get(split)
        if not sizes:
            continue
        gb = sum(sizes) / 2**30
        total += sum(sizes)
        print(f"  {split:12} {len(sizes):4} files   {gb:8.2f} GB")
    print(f"  {'TOTAL':12} {'':4}         {total / 2**30:8.2f} GB\n")

    if not args.get:
        print("sizes only — pass --get transcripts [dev test train] to download")
        return

    accents = args.accents or ["*"]
    patterns = []
    for s in args.get:
        if s == "transcripts":
            patterns += ["transcripts/*", "transcripts/**"]
        else:
            patterns += [f"audio/{a}/{s}/*" for a in accents]

    def wanted(fname: str) -> bool:
        if split_of(fname) not in args.get:
            return False
        return args.accents is None or fname.split("/")[1] in args.accents \
            or fname.startswith("transcripts/")

    selected = [f.rfilename for f in info.siblings if wanted(f.rfilename)]
    check_path_budget(args.dest, selected)
    want = sum(f.size or 0 for f in info.siblings if wanted(f.rfilename)) / 2**30
    print(f"downloading {', '.join(args.get)} (~{want:.2f} GB) to {args.dest}")
    print("resumable: re-run the same command after an interruption.\n")

    path = snapshot_download(REPO, repo_type="dataset", revision=revision,
                             local_dir=args.dest, allow_patterns=patterns,
                             max_workers=4)
    print(f"\ndone: {path}")
    manifest = Path(args.dest) / "FETCHED.json"
    prev = json.loads(manifest.read_text()) if manifest.exists() else {"splits": []}
    prev["repo"], prev["revision"] = REPO, revision
    prev["splits"] = sorted(set(prev["splits"]) | set(args.get))
    manifest.write_text(json.dumps(prev, indent=2))


if __name__ == "__main__":
    main()
