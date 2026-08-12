"""Transcriber interface. One implementation per model family runtime."""

import threading
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Transcription:
    text: str
    # v3-style auto-detecting models: record what language the model decided it heard.
    # An accent triggering wrong-language detection is a fairness finding (EVAL_SPEC §5).
    detected_language: str | None = None
    meta: dict = field(default_factory=dict)


class SerializedInference:
    """Mixin: one model instance serves one inference call at a time.

    None of these runtimes is thread-safe. The model object carries mutable
    per-call state that a second concurrent caller corrupts:
      * NeMo RNNT `transcribe()` freezes the encoder on entry and calls
        `unfreeze(partial=True)` on exit — an interleaved second exit raises
        "Cannot unfreeze partially without first freezing the module".
      * NeMo SALM `generate()` detaches `llm.model.embed_tokens` to splice in
        audio embeddings — a concurrent caller sees the model mid-swap and
        raises "'Qwen3Model' object has no attribute 'embed_tokens'".
      * The moonshine-voice ONNX Transcriber is a single-stream C++ object.
    Both NeMo failures were observed in the 2026-08-11 efficiency runs, where
    every worker past the first died on its first request (EVAL_SPEC changelog).

    Serializing is not merely defensive: one model instance on one GPU *is* a
    serial server, so latency measured under this lock is real queueing delay at
    that offered load, which is what §4.2 asks for. Applied to every backend so
    the closed-loop numbers describe one identical service model across models.
    """

    def _init_inference_lock(self) -> None:
        self._inference_lock = threading.RLock()


class Transcriber(Protocol):
    name: str
    repo_id: str
    revision: str

    def transcribe(self, wav_paths: list[str]) -> list[Transcription]:
        """Transcribe 16 kHz mono wav files. Greedy/default decoding per EVAL_SPEC §5."""
        ...
