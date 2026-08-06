"""EdAcc cleaning rules + normalizer golden test (EVAL_SPEC §4.3).

The golden file freezes normalizer behavior: generated once from the vendored
normalizer, committed, and any future change to normalization breaks the test.
"""

import json
from pathlib import Path

import pytest

from asr_fairness_audit.normalize import is_excluded, normalize, strip_event_tags

GOLDEN = Path(__file__).parent / "golden_normalizer.json"

SAMPLES = [
    "OKAY NOW FOR A REGULAR CONVERSATION SO UH WOULD YOU RATHER GO TO THE BEACH",
    "I MEAN OBVIOUSLY BUT IT'S BEEN I'D RATHER GO RIGHT NOW",
    "YEAH THE WEATHER'S ACTUALLY QUITE NICE",
    "C ELEVEN DASH P ONE",
    "It's twenty-two degrees outside, isn't it?",
    "Dr. Smith paid $3.50 on July 4th",
    "umm... well, you know, like, THAT'S fine!!!",
    "one hundred and one dalmatians",
    "I'm gonna wanna gotta go",
    "co-operate colour flavour theatre",
    "Mm-hmm.",
    "  double   spaces   and\ttabs  ",
    "HE'S AN ABSOLUTE SPECIMEN HE SHOULD NOT BE LET OUT OF THE HOUSE",
    "ESPECIALLY IN THIS TIME STAYING INSIDE",
    "WHAT'S THE QUESTION",
    "I think it's quite an American thing",
    "3 o'clock on the 21st of September, 1999",
    "she said ''oh let's watch die hard'' and left",
    "NO NEVER AGAIN I KNOW YOU'RE A BIG PROMO",
    "the quick brown fox jumps over the lazy dog",
]


def test_exclusion_rules():
    assert is_excluded("IGNORE_TIME_SEGMENT_IN_SCORING") == "ignore_segment"
    assert is_excluded("  IGNORE_TIME_SEGMENT_IN_SCORING  ") == "ignore_segment"
    assert is_excluded("THE DUMPLING <FOREIGN>") == "foreign"
    assert is_excluded("<FOREIGN> IS LIKE MORE POLISH") == "foreign"
    assert is_excluded("A NORMAL SENTENCE") is None


def test_strip_event_tags():
    assert strip_event_tags("YEAH <OVERLAP> OF COURSE").split() == ["YEAH", "OF", "COURSE"]
    assert strip_event_tags("OH MY GOD <DTMF>").split() == ["OH", "MY", "GOD"]
    assert strip_event_tags("ANYWAY <LAUGH> YEAH").split() == ["ANYWAY", "YEAH"]


def test_normalizer_invariants():
    for s in SAMPLES:
        out = normalize(s)
        assert out == out.lower()
        assert "  " not in out
        assert normalize(out) == out, f"not idempotent: {s!r} -> {out!r} -> {normalize(out)!r}"


def test_normalizer_golden():
    outputs = {s: normalize(s) for s in SAMPLES}
    if not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(outputs, indent=2))
        pytest.skip("golden file created — commit tests/golden_normalizer.json and rerun")
    golden = json.loads(GOLDEN.read_text())
    assert outputs == golden, "normalizer behavior changed — this must never happen after freeze"
