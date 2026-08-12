"""One-off: backfill §5/§7 provenance into efficiency.json files that predate it.

    python scripts/stamp_efficiency_provenance.py --list
    python scripts/stamp_efficiency_provenance.py --model parakeet-tdt-0.6b-v3 ...

MUST be run from the same environment that produced the run being stamped —
it captures versions from the *current* interpreter. The 2026-08-11 runs split
across two environments (six under WSL2/nemo-env, Moonshine on native Windows
Python), so this is run once per environment with the matching --model list.

The stamp is marked `backfilled: true`. These versions were read after the fact
from an environment asserted to be unchanged since the run, which is weaker
evidence than a value written by the run itself — the flag says so rather than
letting a post-hoc reading pass as a measurement. Runs from now on record their
own provenance and will be skipped by this script.
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit import MODELS  # noqa: E402
from asr_fairness_audit.provenance import run_provenance  # noqa: E402

RESULTS = ROOT / "results"


def has_provenance(doc: dict) -> bool:
    return "harness" in doc.get("env", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="*", default=None, choices=list(MODELS), metavar="MODEL",
                    help="models to stamp; default is every unstamped efficiency.json")
    ap.add_argument("--list", action="store_true", help="show status and exit")
    ap.add_argument("--force", action="store_true", help="restamp files that already carry provenance")
    args = ap.parse_args()

    files = sorted(RESULTS.glob("*/efficiency.json"))
    if args.model:
        wanted = set(args.model)
        files = [f for f in files if f.parent.name in wanted]
    if not files:
        sys.exit("no matching efficiency.json files")

    if args.list:
        for f in sorted(RESULTS.glob("*/efficiency.json")):
            doc = json.loads(f.read_text())
            state = "stamped" if has_provenance(doc) else "MISSING"
            src = doc.get("env", {}).get("platform", "?")
            print(f"{f.parent.name:30} {state:8} {src[:28]}")
        return

    prov = run_provenance()
    prov["backfilled"] = True
    prov["backfilled_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    prov["backfill_note"] = (
        "versions captured post-hoc from the interpreter that produced the run; "
        "environment asserted unchanged since. Runs after this date self-record.")

    print(f"stamping from: {prov['executable']}")
    print(f"  python {prov['python']} on {prov['platform']}")
    print(f"  harness {prov['harness']['commit'][:12]} dirty={prov['harness']['dirty']}")
    print(f"  libs: {', '.join(f'{k}=={v}' for k, v in prov['libraries'].items())}\n")

    for f in files:
        doc = json.loads(f.read_text())
        if has_provenance(doc) and not args.force:
            print(f"skip   {f.parent.name} (already stamped)")
            continue
        # Sanity: refuse to stamp a run produced on a different OS than this one.
        ran_on = doc.get("env", {}).get("platform", "")
        here = prov["platform"].split("-")[0]
        if ran_on and not ran_on.startswith(here):
            print(f"REFUSE {f.parent.name}: run recorded on {ran_on.split('-')[0]}, "
                  f"stamping from {here} — wrong environment")
            continue
        doc.setdefault("env", {}).update(prov)
        f.write_text(json.dumps(doc, indent=2))
        print(f"stamp  {f.parent.name}")


if __name__ == "__main__":
    main()
