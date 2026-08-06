"""transformers-pipeline backend (Whisper family, Distil-Whisper, Moonshine)."""

import torch
from transformers import pipeline

from .base import Transcription


class HFTranscriber:
    def __init__(self, repo_id: str, revision: str, language: str | None = None,
                 sdpa: bool = False, device: str = "cuda:0", batch_size: int = 8):
        self.name = repo_id.split("/")[-1]
        self.repo_id = repo_id
        self.revision = revision
        self.batch_size = batch_size
        self._generate_kwargs = {"language": language} if language else {}
        model_kwargs = {"attn_implementation": "sdpa"} if sdpa else {}
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=repo_id,
            revision=revision,
            dtype=torch.bfloat16,
            device=device,
            model_kwargs=model_kwargs,
        )

    def transcribe(self, wav_paths: list[str]) -> list[Transcription]:
        outputs = self._pipe(
            list(wav_paths),
            batch_size=self.batch_size,
            generate_kwargs=self._generate_kwargs or None,
        )
        return [Transcription(text=o["text"]) for o in outputs]
