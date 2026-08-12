"""Run provenance: harness commit + runtime library versions (EVAL_SPEC §5, §7).

§7 requires every results file to carry the git commit of the harness that
produced it; §5 requires torch / nemo / transformers / CUDA versions recorded
per run. `pins.json` records those once, at pin time, on one machine — that is
not the same claim. This project runs NeMo models under WSL2 and the Moonshine
ONNX runtime on native Windows, with different interpreters and different torch
builds, so the environment has to be captured per run, not per project.

Versions come from installed package metadata rather than by importing the
packages: importing torch to read `__version__` costs seconds and pulls CUDA
context into a process that may not want one.
"""

import subprocess
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).parents[2]

# Distribution names (not import names) worth recording. Absent ones are skipped:
# a Windows run has no nemo_toolkit, a WSL run has no moonshine-voice.
TRACKED = [
    "torch", "transformers", "datasets", "huggingface_hub", "nemo_toolkit",
    "moonshine-voice", "onnxruntime", "soundfile", "soxr", "numpy", "jiwer",
]


def harness_commit() -> dict:
    """Commit of the harness, plus whether the tree was dirty when it ran.

    A dirty tree means the commit alone does not identify the code that produced
    the numbers, so the flag travels with it rather than being quietly dropped.
    """
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except Exception:
        return {"commit": "unknown", "dirty": None}


def library_versions() -> dict:
    out = {}
    for dist in TRACKED:
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return out


def cuda_versions() -> dict:
    """CUDA as torch sees it, plus the driver. Empty on CPU-only runs (Moonshine)."""
    out = {}
    try:
        import torch
        out["torch_cuda"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    try:
        out["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:
        pass
    return out


def run_provenance() -> dict:
    """Everything §5/§7 want recorded about *how* a run was produced."""
    import platform
    import sys
    return {
        "harness": harness_commit(),
        "libraries": library_versions(),
        "cuda": cuda_versions(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }
