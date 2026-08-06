from asr_fairness_audit.metrics import Utterance, bootstrap_ci, evaluate


def _mk(ref, hyp, group, speaker):
    return Utterance(ref=ref, hyp=hyp, group=group, speaker=speaker)


def test_perfect_transcription():
    utts = [_mk("hello world", "hello world", "g1", "s1"),
            _mk("a b c", "a b c", "g2", "s2")]
    r = evaluate(utts)
    assert r["micro_wer"] == 0.0
    assert r["gap_max_minus_min"] == 0.0


def test_known_wer():
    # g1: 1 sub in 2 words -> 0.5 ; g2: 0 errors in 3 words -> 0.0
    utts = [_mk("hello world", "hello word", "g1", "s1"),
            _mk("a b c", "a b c", "g2", "s2")]
    r = evaluate(utts)
    assert r["per_group"]["g1"]["wer"] == 0.5
    assert r["per_group"]["g2"]["wer"] == 0.0
    assert r["micro_wer"] == 1 / 5
    assert r["macro_wer"] == 0.25
    assert r["worst_group"] == "g1"
    assert r["gap_max_minus_min"] == 0.5


def test_micro_vs_macro_diverge_on_skew():
    # Big easy group + small hard group: micro is dragged down, macro is not.
    utts = [_mk("w " * 50, "w " * 50, "big", f"s{i}") for i in range(5)]
    utts.append(_mk("x y", "a b", "small", "s99"))
    r = evaluate(utts)
    assert r["micro_wer"] < 0.05
    assert r["macro_wer"] == 0.5


def test_empty_ref_counts_insertions():
    r = evaluate([_mk("", "spurious words", "g", "s"),
                  _mk("ok", "ok", "g", "s")])
    assert r["per_group"]["g"]["wer"] == 2 / 1  # 2 insertions / 1 ref word


def test_bootstrap_deterministic():
    utts = [_mk("a b c d", "a b c x", "g1", f"s{i}") for i in range(4)]
    utts += [_mk("a b c d", "a b c d", "g2", f"t{i}") for i in range(4)]
    r1 = bootstrap_ci(utts, n_boot=50)
    r2 = bootstrap_ci(utts, n_boot=50)
    assert r1 == r2  # same seed, same result
    assert r1["ci_low"] <= r1["point"] <= r1["ci_high"]
