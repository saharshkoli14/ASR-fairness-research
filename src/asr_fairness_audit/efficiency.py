"""Efficiency measurement primitives (EVAL_SPEC §4.2).

Separate from accuracy measurement by design: accuracy runs are batch-1 and
output-canonical; efficiency runs may batch and their transcripts are never scored.

Reports tail latency (p50/p95/p99) under closed-loop concurrent load, never
zero-load averages, plus RTFx, peak VRAM, and a thermal/clock record.
"""

import math
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field


def nvidia_query(fields: str) -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        return [p.strip() for p in out.split(",")]
    except Exception:
        return []


# A throttling verdict needs enough 1 Hz samples for the start/end means (each a
# fifth of the run) to be stable. Below this the comparison is DVFS jitter: the
# 2026-08-11 Parakeet runs timed batch phases in 10-18 s, and adjacent samples
# swing ±600 MHz — one phase produced a "-98.7% drop" (the clock rose). Fast
# models are exactly where the check loses power, so the verdict is withheld
# rather than reported as a coin flip. See EVAL_SPEC changelog.
MIN_THROTTLE_SAMPLES = 30


@dataclass
class GpuSampler:
    """Samples clock, temperature, power and memory at 1 Hz during a run (EVAL_SPEC §4.2)."""
    interval: float = 1.0
    samples: list = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def _loop(self):
        while not self._stop.is_set():
            v = nvidia_query("clocks.sm,temperature.gpu,power.draw,memory.used")
            if len(v) == 4:
                try:
                    self.samples.append({"sm_mhz": float(v[0]), "temp_c": float(v[1]),
                                         "power_w": float(v[2]), "mem_mib": float(v[3]),
                                         "t": time.time()})
                except ValueError:
                    pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        if not self.samples:
            return {"sampled": False}
        clk = [s["sm_mhz"] for s in self.samples]
        n = max(1, len(clk) // 5)
        start, end = statistics.mean(clk[:n]), statistics.mean(clk[-n:])
        enough = len(clk) >= MIN_THROTTLE_SAMPLES
        return {
            "sampled": True,
            "n_samples": len(self.samples),
            "sm_clock_start_mhz": round(start, 1),
            "sm_clock_end_mhz": round(end, 1),
            "sm_clock_drop_pct": round(100 * (start - end) / start, 1) if start else 0.0,
            # None = indeterminate, which is not the same as False. Kept distinct so a
            # phase too short to judge can never be summarised as "no throttling".
            "throttled": bool(start and (start - end) / start > 0.10) if enough else None,
            "throttle_verdict": "measured" if enough else
                                f"indeterminate: {len(clk)} samples < {MIN_THROTTLE_SAMPLES}",
            "temp_max_c": max(s["temp_c"] for s in self.samples),
            "power_max_w": max(s["power_w"] for s in self.samples),
            "mem_max_mib": max(s["mem_mib"] for s in self.samples),
        }


def _gpu_blocks(result: dict) -> list[dict]:
    blocks = [v for k, v in result.items() if k.startswith("gpu_") and isinstance(v, dict)]
    blocks += [v["gpu"] for v in result.get("latency", {}).values() if isinstance(v.get("gpu"), dict)]
    return blocks


def recompute_throttle_verdicts(result: dict) -> dict:
    """Apply MIN_THROTTLE_SAMPLES to an already-written result file.

    Only the derived verdict changes; the sampled clocks, temperatures and power
    are untouched. Runs written before the threshold existed reported a verdict
    on as few as 10 samples.
    """
    for b in _gpu_blocks(result):
        if not b.get("sampled") or "n_samples" not in b:
            continue
        if b.get("throttled_not_applicable"):   # CPU run: already voided
            continue
        if b["n_samples"] < MIN_THROTTLE_SAMPLES:
            b["throttled"] = None
            b["throttle_verdict"] = (f"indeterminate: {b['n_samples']} samples "
                                     f"< {MIN_THROTTLE_SAMPLES}")
        else:
            b.setdefault("throttle_verdict", "measured")
    return result


def mark_thermal_not_applicable(result: dict) -> dict:
    """Void the GPU thermal record on runs whose execution device is not the GPU.

    The §4.2 thermal protocol assumes GPU execution. On the CPU-only Moonshine
    ONNX path the sampler still reads the idle GPU, whose SM clock wanders
    between its 210 MHz idle floor and brief boosts — and a fall back to idle
    reads as a >10% "clock drop", i.e. throttling. The 2026-08-11 Moonshine run
    flagged throttling on two phases at 5-13 W draw, which is an idle card.

    The sampled values are kept (they are real, and they document that the GPU
    was idle); only the derived `throttled` verdict is voided, since it answers
    a question about a device that was not doing the work.
    """
    note = ("execution device is CPU; GPU sampled but idle. Clock drop reflects the "
            "card returning to its idle floor, not thermal throttling of the work.")
    for b in _gpu_blocks(result):
        if b.get("sampled"):
            b["throttled"] = None
            b["throttled_not_applicable"] = note
    result["thermal_protocol"] = {
        "applies": False,
        "reason": note,
        "cpu_thermal_sampled": False,
        "consequence": "thermal steady state is unverified for this run (EVAL_SPEC §4.2).",
    }
    return result


def percentiles(xs: list[float]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)

    def pct(p):
        # nearest-rank method: rank = ceil(p/100 * n), 1-indexed. Stated explicitly
        # so the percentile definition is unambiguous in the writeup.
        k = max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))
        return s[k]

    return {"p50": round(pct(50), 4), "p95": round(pct(95), 4), "p99": round(pct(99), 4),
            "mean": round(statistics.fmean(s), 4), "min": round(s[0], 4), "max": round(s[-1], 4),
            "n": len(s)}


@dataclass
class LoadResult:
    """Outcome of one closed-loop run. `errors` must be empty for the run to count."""
    latencies: list[float]
    errors: list[str]
    n_requests: int

    @property
    def complete(self) -> bool:
        return not self.errors and len(self.latencies) == self.n_requests


def closed_loop(fn, items: list, concurrency: int, strict: bool = True) -> LoadResult:
    """Run fn(item) with `concurrency` workers pulling from a shared queue.

    Closed-loop: each worker issues its next request only after the previous
    completes, so measured latency reflects queueing at that load level.

    A raising request is counted as an error and its latency is discarded (a
    failure time is not a service time), but the worker stays alive and pulls
    the next item — otherwise a single exception silently removes a worker and
    the run reports a *lower* offered load than it claims. With strict=True
    (default) any error fails the run rather than producing a partial sweep.
    This is not hypothetical: on 2026-08-11 the NeMo backends lost every worker
    past the first, and c=2/c=4 latencies were written out as if complete
    (n=119/117 of 120). See EVAL_SPEC changelog.
    """
    lock = threading.Lock()
    idx = [0]
    lat: list[float] = []
    errors: list[str] = []

    def worker():
        while True:
            with lock:
                if idx[0] >= len(items):
                    return
                i = idx[0]
                idx[0] += 1
            t0 = time.perf_counter()
            try:
                fn(items[i])
            except Exception as e:  # noqa: BLE001 - recorded, not swallowed
                with lock:
                    errors.append(f"item {i}: {type(e).__name__}: {e}")
                continue
            dt = time.perf_counter() - t0
            with lock:
                lat.append(dt)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    res = LoadResult(latencies=lat, errors=errors, n_requests=len(items))
    if strict and not res.complete:
        raise RuntimeError(
            f"closed_loop(concurrency={concurrency}): {len(errors)} of {len(items)} "
            f"requests failed; latency percentiles would be measured at an unknown "
            f"offered load. First: {errors[0] if errors else 'n/a'}"
        )
    return res
