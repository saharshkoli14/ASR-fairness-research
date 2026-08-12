"""Regression tests for the closed-loop load harness (EVAL_SPEC §4.2).

Guards the 2026-08-11 failure: the NeMo runtimes are not thread-safe, every
closed-loop worker past the first died on its first request, and the harness
wrote the resulting partial sweep out as if it were complete.

The fake model below reproduces NeMo's RNNT contract exactly — freeze on entry,
unfreeze on exit, one shared flag — which is what makes concurrent calls corrupt
each other. Overlap is forced with a Barrier rather than a sleep so the test is
deterministic: unguarded, both threads meet at the barrier and the race fires;
guarded, the barrier can never be met, times out, and the run is clean.
"""

import threading

import pytest

from asr_fairness_audit.backends.base import SerializedInference
from asr_fairness_audit.efficiency import closed_loop, percentiles

BARRIER_TIMEOUT = 0.05


class FakeFreezeUnfreezeModel:
    """Mimics NeMo RNNT `transcribe()`: freeze on entry, unfreeze on exit.

    Interleaved A-enter, B-enter, A-exit, B-exit leaves B unfreezing a module
    that is no longer frozen — the real ValueError from nemo/core/classes/module.py.
    """

    def __init__(self, barrier: threading.Barrier):
        self._frozen = False
        self._barrier = barrier
        self.calls = 0

    def transcribe(self, paths):
        self._frozen = True                     # _transcribe_on_begin
        try:
            self._barrier.wait(timeout=BARRIER_TIMEOUT)
        except threading.BrokenBarrierError:
            pass                                # no second caller: serialized
        if not self._frozen:                    # _transcribe_on_end
            raise ValueError(
                "Cannot unfreeze partially without first freezing the module with `freeze()`")
        self._frozen = False
        self.calls += 1
        return ["ok"] * len(paths)


class UnguardedTranscriber:
    def __init__(self, model):
        self._model = model

    def transcribe(self, paths):
        return self._model.transcribe(paths)


class LockedTranscriber(SerializedInference):
    def __init__(self, model):
        self._model = model
        self._init_inference_lock()

    def transcribe(self, paths):
        with self._inference_lock:
            return self._model.transcribe(paths)


def _run(transcriber_cls, concurrency, n_items=8, **kw):
    barrier = threading.Barrier(2)
    model = FakeFreezeUnfreezeModel(barrier)
    t = transcriber_cls(model)
    items = [f"clip{i}.wav" for i in range(n_items)]
    return closed_loop(lambda p: t.transcribe([p]), items, concurrency=concurrency, **kw), model


# --- the bug ---------------------------------------------------------------

def test_unguarded_backend_races_under_concurrency():
    """Without serialization the shared model corrupts itself. This is the 08-11 bug."""
    res, _ = _run(UnguardedTranscriber, concurrency=2, strict=False)
    assert res.errors, "expected the freeze/unfreeze race to fire"
    assert "Cannot unfreeze partially" in res.errors[0]
    assert not res.complete


def test_failed_request_does_not_kill_its_worker():
    """Every item must still be attempted: latencies + errors account for all of them.

    Previously an exception unwound the worker thread, so the run continued at a
    lower offered load than it reported and the missing items vanished silently.
    """
    n = 8
    res, _ = _run(UnguardedTranscriber, concurrency=2, n_items=n, strict=False)
    assert len(res.latencies) + len(res.errors) == n
    assert res.n_requests == n


def test_strict_mode_refuses_to_report_a_partial_sweep():
    with pytest.raises(RuntimeError, match="requests failed"):
        _run(UnguardedTranscriber, concurrency=2)


# --- the fix ---------------------------------------------------------------

@pytest.mark.parametrize("concurrency", [1, 2, 4])
def test_serialized_backend_is_clean_at_every_concurrency(concurrency):
    n = 8
    res, model = _run(LockedTranscriber, concurrency=concurrency, n_items=n)
    assert res.errors == []
    assert res.complete
    assert len(res.latencies) == n == model.calls


def test_lock_is_per_instance_not_shared():
    """Two transcribers must not block each other; the lock guards one model's state."""
    a = LockedTranscriber(FakeFreezeUnfreezeModel(threading.Barrier(2)))
    b = LockedTranscriber(FakeFreezeUnfreezeModel(threading.Barrier(2)))
    assert a._inference_lock is not b._inference_lock


def test_lock_is_reentrant():
    """RLock: a backend may call a locked helper from inside a locked transcribe()."""
    t = LockedTranscriber(FakeFreezeUnfreezeModel(threading.Barrier(2)))
    with t._inference_lock:
        with t._inference_lock:
            pass


# --- percentile contract ---------------------------------------------------

def test_percentiles_nearest_rank():
    xs = [float(i) for i in range(1, 101)]  # 1..100
    p = percentiles(xs)
    assert (p["p50"], p["p95"], p["p99"]) == (50.0, 95.0, 99.0)
    assert p["n"] == 100 and p["min"] == 1.0 and p["max"] == 100.0


def test_percentiles_empty():
    assert percentiles([]) == {}
