"""Vendor the Whisper EnglishTextNormalizer into the package (EVAL_SPEC §4.3).

Downloads the normalizer source from openai/whisper at the current main commit,
records that commit in _vendor/VENDOR_INFO.json, and writes the files into
src/asr_fairness_audit/_vendor/. Run once, commit the result. A library upgrade
can then never silently change our text normalization.

    python scripts/vendor_normalizer.py
"""

import json
import sys
import urllib.request
from pathlib import Path

VENDOR_DIR = Path(__file__).parents[1] / "src" / "asr_fairness_audit" / "_vendor"
FILES = ["whisper/normalizers/english.py", "whisper/normalizers/basic.py", "whisper/normalizers/english.json"]
API_COMMIT = "https://api.github.com/repos/openai/whisper/commits/main"
RAW = "https://raw.githubusercontent.com/openai/whisper/{sha}/{path}"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def main():
    info_file = VENDOR_DIR / "VENDOR_INFO.json"
    if info_file.exists() and "--force" not in sys.argv:
        sys.exit(f"Already vendored ({json.loads(info_file.read_text())['commit'][:12]}). Use --force to re-vendor.")

    sha = json.loads(fetch(API_COMMIT))["sha"]
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    (VENDOR_DIR / "__init__.py").write_text("")
    for path in FILES:
        data = fetch(RAW.format(sha=sha, path=path))
        name = path.split("/")[-1]
        (VENDOR_DIR / name).write_bytes(data)
        print(f"vendored {name} ({len(data):,} bytes)")

    # english.py does `from .basic import ...` — our flat vendor dir keeps that import valid.
    info_file.write_text(json.dumps({"source": "openai/whisper", "commit": sha, "files": FILES}, indent=2))
    print(f"\nSource commit: {sha}\nCommit the _vendor/ directory.")


if __name__ == "__main__":
    main()
