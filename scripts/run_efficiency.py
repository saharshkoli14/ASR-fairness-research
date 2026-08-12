"""Efficiency benchmark (EVAL_SPEC §4.2): RTFx, tail latency under load, peak VRAM.

    python scripts/run_efficiency.py --model parakeet-tdt-0.6b-v3

Protocol:
  * fixed sample of real EdAcc test audio (same clips for every model, seed-fixed)
  * 5-minute warmup at load before any timing (thermal steady state)
  * RTFx measured at batch 1 and at the largest batch that fits
  * latency measured closed-loop at concurrency 1, 2, 4 — p50/p95/p99, never averages.
    One model instance serves one request at a time (backends serialize on a lock),
    so these are queueing latencies under the service model a single GPU provides.
    Any failed request aborts the run: a short sweep must never look like a complete one.
  * GPU clock/temp/power sampled at 1 Hz throughout; >10% clock drop flags throttling
  * peak VRAM from torch allocator and from nvidia-smi

Transcripts produced here are NEVER scored — accuracy comes from run_audit.py only.
Run on an idle machine, AC power, performance profile.
"""

import argparse
import io
import json
import platform
import random
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit import MODELS, get_transcriber, load_pins  # noqa: E402
from asr_fairness_audit.data.edacc import load_edacc  # noqa: E402
from asr_fairness_audit.efficiency import (GpuSampler, closed_loop,  # noqa: E402
                                           mark_thermal_not_applicable, nvidia_query,  # noqa: E402
                                           percentiles, recompute_throttle_verdicts)
from asr_fairness_audit.provenance import run_provenance  # noqa: E402

SEED = 3407
N_CLIPS = 120          # timing sample
WARMUP_SECONDS = 300   # 5 min at load before timing (EVAL_SPEC §4.2)
CONCURRENCY = [1, 2, 4]
MAX_CLIP_S = 30.0      # cap so one 536 s outlier doesn't dominate throughput


def build_sample(tmp: Path, pins: dict) -> tuple[list[str], float]:
    """Fixed, seed-stable sample of real test audio. Same clips for every model."""
    rows = load_edacc("test", pins).rows
    rng = random.Random(SEED)
    idx = [i for i in range(len(rows))]
    rng.shuffle(idx)
    paths, total = [], 0.0
    for i in idx:
        b = rows[i]["audio"]["bytes"]
        dur = sf.info(io.BytesIO(b)).duration
        if dur > MAX_CLIP_S or dur < 1.0:
            continue
        p = tmp / f"{i}.wav"
        p.write_bytes(b)
        paths.append(str(p))
        total += dur
        if len(paths) >= N_CLIPS:
            break
    return paths, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS.keys())
    ap.add_argument("--warmup", type=int, default=WARMUP_SECONDS)
    ap.add_argument("--concurrency", type=int, nargs="*", default=CONCURRENCY)
    args = ap.parse_args()

    pins = load_pins()
    out_dir = ROOT / "results" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # §5/§7: the commit and the exact library stack that produced these numbers.
    # Recorded per run, not per project — NeMo runs under WSL2 and Moonshine on
    # native Windows, so pins.json's single env_at_pin_time does not describe them.
    env = {
        "gpu": nvidia_query("name,power.limit,driver_version"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        **run_provenance(),
    }
    print(f"GPU: {env['gpu']}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        clips, audio_seconds = build_sample(tmp, pins)
        print(f"sample: {len(clips)} clips, {audio_seconds / 60:.1f} min of audio")

        t = get_transcriber(args.model, pins)
        one = lambda p: t.transcribe([p])  # noqa: E731

        print(f"warmup: {args.warmup}s at load...")
        t0 = time.time()
        i = 0
        while time.time() - t0 < args.warmup:
            one(clips[i % len(clips)])
            i += 1
        print(f"  {i} warmup requests")

        # Execution device is NOT uniform across backends: the moonshine-voice ONNX
        # wheel is CPU-only, while HF/NeMo backends run on CUDA. RTFx and latency are
        # therefore not comparable across the CPU/GPU boundary — record it explicitly
        # so the results table can say so rather than implying a like-for-like race.
        backend = MODELS[args.model][1]
        device = "cpu-onnx" if backend == "moonshine" else "cuda"
        result = {"model": args.model, "repo_id": MODELS[args.model][0],
                  "model_revision": pins["models"][MODELS[args.model][0]],
                  "backend": backend, "device": device,
                  "env": env, "n_clips": len(clips),
                  "audio_seconds": round(audio_seconds, 1),
                  "warmup_seconds": args.warmup, "seed": SEED}

        # --- RTFx at batch 1 (sequential, no concurrency) ---
        try:
            import torch
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            torch = None

        with GpuSampler() as smp:
            t0 = time.perf_counter()
            for p in clips:
                one(p)
            wall = time.perf_counter() - t0
        result["rtfx_batch1"] = round(audio_seconds / wall, 2)
        result["batch1_wall_s"] = round(wall, 1)
        result["gpu_batch1"] = smp.summary()
        if torch is not None and torch.cuda.is_available():
            # Headline VRAM figure: batch 1, before any batched run can spill and
            # inflate the allocator high-water mark.
            result["peak_vram_batch1_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        print(f"RTFx (batch 1): {result['rtfx_batch1']}  [{wall:.1f}s wall]")

        # --- RTFx batched (efficiency-only; outputs never scored) ---
        for bs in (4, 8, 16):
            if not hasattr(t, "batch_size"):
                break
            prev = t.batch_size
            t.batch_size = bs
            try:
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                with GpuSampler() as smp:
                    t0 = time.perf_counter()
                    for k in range(0, len(clips), bs):
                        t.transcribe(clips[k:k + bs])
                    wall = time.perf_counter() - t0
                result[f"rtfx_batch{bs}"] = round(audio_seconds / wall, 2)
                result[f"gpu_batch{bs}"] = smp.summary()
                if torch is not None and torch.cuda.is_available():
                    vram = round(torch.cuda.max_memory_reserved() / 2**30, 2)
                    result[f"peak_vram_batch{bs}_gb"] = vram
                    # >7.5 GB usable on this card: beyond it CUDA silently spills to
                    # system RAM over PCIe and throughput collapses (a real deployment
                    # failure mode, reported rather than hidden).
                    result[f"spilled_batch{bs}"] = vram > 7.5
                print(f"RTFx (batch {bs}): {result[f'rtfx_batch{bs}']}"
                      + (f"  [VRAM {result[f'peak_vram_batch{bs}_gb']} GB"
                         f"{' — SPILLED' if result.get(f'spilled_batch{bs}') else ''}]"
                         if f"peak_vram_batch{bs}_gb" in result else ""))
            except Exception as e:
                result[f"rtfx_batch{bs}"] = f"failed: {type(e).__name__}"
                print(f"RTFx (batch {bs}): failed ({type(e).__name__})")
            finally:
                t.batch_size = prev

        # --- latency under closed-loop concurrency ---
        # Every backend serializes model calls on a per-instance lock
        # (backends.base.SerializedInference), so these are queueing latencies
        # against one model instance — the same service model for every model.
        result["latency"] = {}
        result["service_model"] = "single instance, requests serialized (EVAL_SPEC §4.2)"
        for c in args.concurrency:
            with GpuSampler() as smp:
                run = closed_loop(one, clips, concurrency=c)  # raises on any failed request
            result["latency"][f"concurrency_{c}"] = {
                **percentiles(run.latencies), "requests_ok": len(run.latencies),
                "requests_issued": run.n_requests, "gpu": smp.summary()}
            p = result["latency"][f"concurrency_{c}"]
            print(f"latency c={c}: p50 {p['p50']:.3f}s  p95 {p['p95']:.3f}s  p99 {p['p99']:.3f}s")

        if torch is not None and torch.cuda.is_available():
            result["peak_vram_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            result["peak_vram_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)

    # The thermal protocol watches the GPU; on the CPU-only ONNX path it watches an
    # idle card, whose return to its idle clock floor reads as a >10% drop.
    if device != "cuda":
        mark_thermal_not_applicable(result)
        print("\nNote: execution device is CPU — GPU thermal record voided, "
              "thermal steady state unverified for this run.")
    else:
        recompute_throttle_verdicts(result)

    (out_dir / "efficiency.json").write_text(json.dumps(result, indent=2))
    thr = [k for k, v in result.items() if isinstance(v, dict) and v.get("throttled")]
    unk = [k for k, v in result.items()
           if isinstance(v, dict) and str(v.get("throttle_verdict", "")).startswith("indeterm")]
    if thr:
        print(f"\nWARNING: thermal throttling detected during: {thr}")
    if unk:
        print(f"\nNote: throttling indeterminate (phase too short to judge) during: {unk}")
    print(f"\nWrote {out_dir / 'efficiency.json'}")


if __name__ == "__main__":
    main()
