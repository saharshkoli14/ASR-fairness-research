"""transformers-pipeline backend (Whisper family, Distil-Whisper, Moonshine).

Audio is decoded with soundfile and resampled to 16 kHz with soxr in our code,
then passed as arrays. This avoids the pipeline's ffmpeg-subprocess path (a
hidden system dependency) and makes resampling identical across all HF models.
All models in the registry expect 16 kHz input.
"""

import numpy as np
import soundfile as sf
import soxr
import torch
from transformers import pipeline

from .base import Transcription

TARGET_SR = 16_000


def load_16k_mono(wav_path: str) -> np.ndarray:
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        data = soxr.resample(data, sr, TARGET_SR)
    return data


class HFTranscriber:
    # batch_size=1 is REQUIRED for accuracy runs: batched padding changes Whisper
    # outputs (6/50 differed on the smoke set), and >30 s utterances only get
    # long-form decoding at batch 1. Do not raise for accuracy. (EVAL_SPEC changelog 2026-08-07.)
    # dtype follows each model card's documented default (EVAL_SPEC §5): fp16 for
    # the Whisper family, fp32 for Moonshine (library default; 245M has no memory need).
    def __init__(self, repo_id: str, revision: str, language: str | None = None,
                 sdpa: bool = False, device: str = "cuda:0", batch_size: int = 1,
                 dtype: str = "float16", pad_to_multiple: int | None = None,
                 chunk_s: float | None = None):
        self.name = repo_id.split("/")[-1]
        self.repo_id = repo_id
        self.revision = revision
        self.batch_size = batch_size
        self.pad_to_multiple = pad_to_multiple
        # chunk_s: fixed-length chunking for models whose HF implementation fails on
        # long inputs (Moonshine Streaming + SDPA: CUDA index assert on long audio).
        # Chunks are transcribed independently and joined with a space. Recorded deviation.
        self.chunk_s = chunk_s
        # Whisper family (language is set): return_timestamps=True enables the model's
        # documented sequential long-form decoding for >30 s utterances (EVAL_SPEC §5
        # "default chunking"). Applied uniformly to ALL utterances so decoding config is
        # identical across the split. It is a PIPELINE-level argument, not a generate kwarg.
        # Moonshine (no language kwarg) is unwindowed and takes neither argument.
        self._return_timestamps = bool(language)
        self._generate_kwargs = {"language": language} if language else {}
        model_kwargs = {"attn_implementation": "sdpa"} if sdpa else {}
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=repo_id,
            revision=revision,
            dtype=getattr(torch, dtype),
            device=device,
            model_kwargs=model_kwargs,
        )

    def _prep(self, path: str) -> dict:
        data = load_16k_mono(path)
        if self.pad_to_multiple and (r := len(data) % self.pad_to_multiple):
            data = np.pad(data, (0, self.pad_to_multiple - r))  # Moonshine: frame-multiple input
        return {"array": data, "sampling_rate": TARGET_SR}

    def _run(self, inputs: list[dict]) -> list[str]:
        kwargs = {"batch_size": self.batch_size}
        if self._return_timestamps:
            kwargs["return_timestamps"] = True
        if self._generate_kwargs:
            kwargs["generate_kwargs"] = self._generate_kwargs
        return [o["text"] for o in self._pipe(inputs, **kwargs)]

    def transcribe(self, wav_paths: list[str]) -> list[Transcription]:
        results: list[Transcription] = []
        for p in wav_paths:
            inp = self._prep(p)
            max_len = int(self.chunk_s * TARGET_SR) if self.chunk_s else None
            if max_len and len(inp["array"]) > max_len:
                arr = inp["array"]
                pieces = [{"array": arr[i:i + max_len], "sampling_rate": TARGET_SR}
                          for i in range(0, len(arr), max_len)]
                texts = self._run(pieces)
                results.append(Transcription(text=" ".join(t.strip() for t in texts),
                                             meta={"chunked": len(pieces)}))
            else:
                results.append(Transcription(text=self._run([inp])[0]))
        return results
