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
    # transformers path retained for the determinism bug report only — NOT reportable (§4.5).
    "moonshine-streaming-medium-hf": ("UsefulSensors/moonshine-streaming-medium", "hf",
                                      {"sdpa": True, "dtype": "float32", "pad_to_multiple": 80,
                                       "chunk_s": 30.0}),
    # Official ONNX runtime — the reportable Moonshine configuration.
    "moonshine-streaming-medium": ("UsefulSensors/moonshine-streaming-medium", "moonshine",
                                   {"arch": "MEDIUM_STREAMING", "language": "en"}),
    "whisper-small": ("openai/whisper-small", "hf", {"language": "en", "dtype": "float16"}),
}


def load_pins(pins_path: str | Path | None = None) -> dict:
    """Load pinned revisions. Every model/dataset load must pass one of these."""
    path = Path(pins_path) if pins_path else Path(__file__).parents[2] / "pins.json"
    if not path.exists():
        raise FileNotFoundError(f"pins.json not found at {path} — run pin.py first. Unpinned loads are banned.")
    return json.loads(path.read_text())


def get_transcriber(name: str, pins: dict, checkpoint: str | None = None):
    """Build a Transcriber for a registered model, loading at its pinned revision.

    `checkpoint` overrides the weights with a local directory (a fine-tuned model
    from §6) while keeping the registry entry's backend and inference kwargs. The
    pinned revision no longer identifies those weights, so callers must record the
    checkpoint path instead — run_audit.py writes it into summary.json.
    """
    repo_id, backend, kwargs = MODELS[name]
    revision = pins["models"][repo_id]
    if checkpoint:
        if backend != "hf":
            raise ValueError(f"--checkpoint is only supported for the hf backend, not {backend!r}")
        # Weights from the checkpoint, tokenizer/feature extractor from the pinned base:
        # snapshot dirs contain no tokenizer, and fine-tuning does not change it.
        kwargs = {**kwargs, "processor_id": repo_id, "processor_revision": revision}
        repo_id, revision = checkpoint, None
    if backend == "hf":
        from .backends.hf import HFTranscriber

        return HFTranscriber(repo_id, revision=revision, **kwargs)
    if backend == "moonshine":
        from .backends.moonshine import MoonshineVoiceTranscriber

        return MoonshineVoiceTranscriber(repo_id, revision=revision, **kwargs)
    from .backends.nemo import NeMoTranscriber

    return NeMoTranscriber(repo_id, revision=revision, **kwargs)
