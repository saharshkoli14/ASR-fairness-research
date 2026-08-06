"""Feasibility check: can each model load and transcribe within 7.5 GB VRAM on the RTX 4060?

Run ONE model per process so VRAM measurements don't contaminate each other:

    python feasibility.py --all            # spawns a fresh subprocess per model
    python feasibility.py --model whisper-large-v3-turbo   # single model (used by --all)

Results append to feasibility_results.json. This measures LOAD + single-forward-pass
feasibility only — it is not the benchmark. Numbers here never appear in the paper.

Notes:
- NeMo models (canary-qwen, parakeet) are expected to FAIL on native Windows.
  NeMo targets Linux; run those two under WSL2 with CUDA. A recorded failure
  here is a valid feasibility result, not a bug in this script.
- Moonshine Streaming prefers flash-attention; we force SDPA and record whether
  that works — that's the point of the test.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / "feasibility_results.json"
VRAM_BUDGET_GB = 7.5

# name -> (repo_id, backend)
MODELS = {
    "whisper-large-v3-turbo": ("openai/whisper-large-v3-turbo", "hf"),
    "distil-large-v3.5": ("distil-whisper/distil-large-v3.5", "hf"),
    "whisper-small": ("openai/whisper-small", "hf"),
    "moonshine-streaming-medium": ("UsefulSensors/moonshine-streaming-medium", "hf-sdpa"),
    "parakeet-tdt-0.6b-v2": ("nvidia/parakeet-tdt-0.6b-v2", "nemo-asr"),
    "parakeet-tdt-0.6b-v3": ("nvidia/parakeet-tdt-0.6b-v3", "nemo-asr"),
    "canary-qwen-2.5b": ("nvidia/canary-qwen-2.5b", "nemo-salm"),
}


def make_test_audio(seconds: float = 10.0, sr: int = 16000):
    """Synthetic speech-band audio. Content is irrelevant; we test load + forward pass."""
    import numpy as np

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sig = 0.1 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    sig += 0.02 * np.random.default_rng(0).standard_normal(sig.shape)
    return sig.astype("float32"), sr


def gpu_env():
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False}
    p = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "gpu": p.name,
        "total_vram_gb": round(p.total_memory / 2**30, 2),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
    }


def run_hf(repo_id: str, force_sdpa: bool):
    import torch
    from transformers import pipeline

    audio, sr = make_test_audio()
    kwargs = {"torch_dtype": torch.bfloat16, "device": "cuda:0"}
    if force_sdpa:
        kwargs["model_kwargs"] = {"attn_implementation": "sdpa"}

    t0 = time.perf_counter()
    pipe = pipeline("automatic-speech-recognition", model=repo_id, **kwargs)
    load_s = time.perf_counter() - t0

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = pipe({"array": audio, "sampling_rate": sr})
    infer_s = time.perf_counter() - t0
    return load_s, infer_s, str(out.get("text", out))[:200]


def run_nemo(repo_id: str, backend: str):
    import torch
    import soundfile as sf

    audio, sr = make_test_audio()
    wav_path = Path(__file__).parent / "_feas_test.wav"
    sf.write(wav_path, audio, sr)

    t0 = time.perf_counter()
    if backend == "nemo-salm":
        from nemo.collections.speechlm2.models import SALM

        model = SALM.from_pretrained(repo_id)
        model = model.to(torch.bfloat16)  # NeMo restores fp32 (~10 GB); card intends bf16 (~5 GB)
    else:
        from nemo.collections.asr.models import ASRModel

        model = ASRModel.from_pretrained(model_name=repo_id)
    model = model.to("cuda").eval()
    load_s = time.perf_counter() - t0

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    if backend == "nemo-salm":
        answer_ids = model.generate(
            prompts=[[{
                "role": "user",
                "content": f"Transcribe the following: {model.audio_locator_tag}",
                "audio": [str(wav_path)],
            }]],
            max_new_tokens=128,
        )
        text = model.tokenizer.ids_to_text(answer_ids[0].cpu())[:200]
    else:
        out = model.transcribe([str(wav_path)])
        text = str(out[0])[:200]
    infer_s = time.perf_counter() - t0
    wav_path.unlink(missing_ok=True)
    return load_s, infer_s, text


def test_one(name: str) -> dict:
    repo_id, backend = MODELS[name]
    rec = {
        "model": name,
        "repo_id": repo_id,
        "backend": backend,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": {},
        "ok": False,
    }
    try:
        import torch

        rec["env"] = gpu_env()
        if not rec["env"].get("cuda"):
            raise RuntimeError("CUDA not available in this environment")

        if backend.startswith("hf"):
            load_s, infer_s, text = run_hf(repo_id, force_sdpa=(backend == "hf-sdpa"))
        else:
            load_s, infer_s, text = run_nemo(repo_id, backend)

        peak_gb = torch.cuda.max_memory_allocated() / 2**30
        reserved_gb = torch.cuda.max_memory_reserved() / 2**30
        rec.update(
            ok=True,
            load_seconds=round(load_s, 1),
            infer_seconds_10s_audio=round(infer_s, 2),
            peak_vram_allocated_gb=round(peak_gb, 2),
            peak_vram_reserved_gb=round(reserved_gb, 2),
            fits_budget=reserved_gb <= VRAM_BUDGET_GB,
            sample_output=text,
        )
    except Exception as e:  # noqa: BLE001 — we want every failure recorded, not raised
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def append_result(rec: dict):
    results = []
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text())
    results.append(rec)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS.keys())
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        for name in MODELS:
            print(f"\n=== {name} (fresh process) ===")
            subprocess.run([sys.executable, __file__, "--model", name], check=False)
        results = json.loads(RESULTS_FILE.read_text())
        print("\n=== SUMMARY ===")
        for r in results:
            if r["ok"]:
                print(f"  OK   {r['model']:28s} peak {r['peak_vram_reserved_gb']:.2f} GB "
                      f"{'(fits)' if r['fits_budget'] else '(OVER BUDGET)'}")
            else:
                print(f"  FAIL {r['model']:28s} {r.get('error', '?')[:100]}")
    elif args.model:
        rec = test_one(args.model)
        append_result(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "env"}, indent=2))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
