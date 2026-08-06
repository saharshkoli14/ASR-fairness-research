"""Text normalization for WER (EVAL_SPEC §4.3): vendored Whisper EnglishTextNormalizer.

Applied identically to references and hypotheses. The vendored copy is the ONLY
normalizer allowed to touch scored text — never import a normalizer from an
installed library at eval time.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

_VENDOR = Path(__file__).parent / "_vendor"

# EdAcc reference cleaning (EVAL_SPEC §4.3). Order matters: exclusion checks run
# on RAW reference text before any stripping.
IGNORE_LITERAL = "IGNORE_TIME_SEGMENT_IN_SCORING"
FOREIGN_TAG = "<FOREIGN>"
_EVENT_TAG = re.compile(r"<[A-Z_]+>")


def is_excluded(raw_ref: str) -> str | None:
    """Return exclusion reason for an EdAcc reference, or None if scoreable."""
    if raw_ref.strip() == IGNORE_LITERAL:
        return "ignore_segment"
    if FOREIGN_TAG in raw_ref:
        return "foreign"
    return None


def strip_event_tags(text: str) -> str:
    """Remove non-speech event tags (<OVERLAP>, <LAUGH>, <DTMF>, ...) from references."""
    return _EVENT_TAG.sub(" ", text)


@lru_cache(maxsize=1)
def get_normalizer():
    try:
        from ._vendor.english import EnglishTextNormalizer
    except ImportError as e:
        raise RuntimeError(
            "Vendored normalizer missing — run scripts/vendor_normalizer.py and commit _vendor/."
        ) from e
    return EnglishTextNormalizer()


def normalize(text: str) -> str:
    return get_normalizer()(text)


def normalize_reference(raw_ref: str) -> str:
    """Full EdAcc reference path: strip event tags, then Whisper normalization."""
    return normalize(strip_event_tags(raw_ref))


def vendor_info() -> dict:
    return json.loads((_VENDOR / "VENDOR_INFO.json").read_text())
