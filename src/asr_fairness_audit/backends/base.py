"""Transcriber interface. One implementation per model family runtime."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Transcription:
    text: str
    # v3-style auto-detecting models: record what language the model decided it heard.
    # An accent triggering wrong-language detection is a fairness finding (EVAL_SPEC §5).
    detected_language: str | None = None
    meta: dict = field(default_factory=dict)


class Transcriber(Protocol):
    name: str
    repo_id: str
    revision: str

    def transcribe(self, wav_paths: list[str]) -> list[Transcription]:
        """Transcribe 16 kHz mono wav files. Greedy/default decoding per EVAL_SPEC §5."""
        ...
