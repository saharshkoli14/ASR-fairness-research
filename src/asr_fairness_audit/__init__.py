"""asr_fairness_audit — accent-fairness disparity audit harness.

Methodology is frozen in EVAL_SPEC.md (v1.0). All loads use pinned revisions
from pins.json; see load_pins().
"""

import json
from pathlib import Path

__version__ = "0.1.0"

# Model registry: short name -> (repo_id, backend, backend kwargs).
# Mirrors EVAL_SPEC.md §1. Backends: "hf" (transformers pipeline) or "nemo".
MODELS = {
    "canary-qwen-2.5b": ("nvidia/canary-qwen-2.5b", "nemo", {"kind": "salm", "cast_bf16": True}),
    "parakeet-tdt-0.6b-v2": ("nvidia/parakeet-tdt-0.6b-v2", "nemo", {"kind": "asr"}),
    "parakeet-tdt-0.6b-v3": ("nvidia/parakeet-tdt-0.6b-v3", "nemo", {"kind": "asr", "track_language": True}),
    "whisper-large-v3-turbo": ("openai/whisper-large-v3-turbo", "hf", {"language": "en", "dtype": "float16"}),
    "distil-large-v3.5": ("distil-whisper/distil-large-v3.5", "hf", {"language": "en", "dtype": "float16"}),
    "moonshine-streaming-medium": ("UsefulSensors/moonshine-streaming-medium", "hf",
                                   {"sdpa": True, "dtype": "float32", "pad_to_multiple": 80}),
    "whisper-small": ("openai/whisper-small", "hf", {"language": "en", "dtype": "float16"}),
}


def load_pins(pins_path: str | Path | None = None) -> dict:
    """Load pinned revisions. Every model/dataset load must pass one of these."""
    path = Path(pins_path) if pins_path else Path(__file__).parents[2] / "pins.json"
    if not path.exists():
        raise FileNotFoundError(f"pins.json not found at {path} — run pin.py first. Unpinned loads are banned.")
    return json.loads(path.read_text())


def get_transcriber(name: str, pins: dict):
    """Build a Transcriber for a registered model, loading at its pinned revision."""
    repo_id, backend, kwargs = MODELS[name]
    revision = pins["models"][repo_id]
    if backend == "hf":
        from .backends.hf import HFTranscriber

        return HFTranscriber(repo_id, revision=revision, **kwargs)
    from .backends.nemo import NeMoTranscriber

    return NeMoTranscriber(repo_id, revision=revision, **kwargs)
