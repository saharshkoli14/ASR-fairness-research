"""Official Moonshine Voice runtime backend (moonshine-voice, ONNX Runtime C++ core).

Used instead of transformers because the transformers implementation of
moonshine_streaming fails the §4.5 determinism gate (identical audio -> different
text depending on process state). This runtime is a separate code path entirely.

Documented behavioural differences vs the transformers path — both recorded in
results and LIMITATIONS.md:
  * VAD segmentation: the runtime splits input into transcript "lines"; we join
    them in time order to form one hypothesis per utterance.
  * max_tokens_per_second (default 6.5): a built-in anti-hallucination-loop
    heuristic that truncates runaway decoding. No Whisper-family model has an
    equivalent, so Moonshine's loop-rate diagnostic is NOT comparable to theirs
    unless this is disabled (see EVAL_SPEC §4.5 note).
"""

import numpy as np

from .base import Transcription
from .hf import TARGET_SR, load_16k_mono


class MoonshineVoiceTranscriber:
    def __init__(self, repo_id: str, revision: str, arch: str = "MEDIUM_STREAMING",
                 language: str = "en", max_tokens_per_second: float | None = None,
                 model_path: str | None = None, batch_size: int = 1):
        import moonshine_voice as mv

        self.name = "moonshine-voice-" + arch.lower()
        self.repo_id = repo_id          # recorded for provenance; weights come from the CDN
        self.revision = revision
        self.batch_size = 1             # runtime is inherently one-stream
        self._mv = mv
        self.arch = getattr(mv.ModelArch, arch)

        if model_path is None:
            model_path = self._resolve_model_path(mv, language)
        self.model_path = str(model_path)

        options = {}
        if max_tokens_per_second is not None:
            options["max_tokens_per_second"] = str(max_tokens_per_second)
        self.options = options

        self._transcriber = mv.Transcriber(
            model_path=self.model_path,
            model_arch=self.arch,
            options=options or None,
        )

    def _resolve_model_path(self, mv, language: str) -> str:
        """Locate downloaded model files; the API surface varies, so try known entry points."""
        for fn_name in ("get_model_path", "get_model_for_language"):
            fn = getattr(mv, fn_name, None)
            if fn is None:
                continue
            for args, kwargs in (((), {"language": language, "model_arch": self.arch}),
                                 ((language, self.arch), {}),
                                 ((), {"model_arch": self.arch}),
                                 ((), {"language": language})):
                try:
                    out = fn(*args, **kwargs)
                except Exception:
                    continue
                # Some helpers return (path, arch)
                if isinstance(out, tuple):
                    out = out[0]
                if out:
                    return out
        raise RuntimeError(
            "Could not resolve Moonshine model path. Run:\n"
            "  python -m moonshine_voice.download --language en --stt --model-arch medium_streaming\n"
            "then pass model_path= explicitly (the download prints its location)."
        )

    def _text_from_transcript(self, transcript) -> str:
        lines = getattr(transcript, "lines", None)
        if lines is None:
            lines = list(transcript) if hasattr(transcript, "__iter__") else []
        parts = []
        for ln in lines:
            t = getattr(ln, "text", None)
            if t:
                parts.append(t.strip())
        return " ".join(parts)

    def transcribe(self, wav_paths: list[str]) -> list[Transcription]:
        out = []
        for p in wav_paths:
            audio = load_16k_mono(p)  # same decode/resample path as the HF backend
            transcript = self._transcriber.transcribe_without_streaming(
                np.asarray(audio, dtype=np.float32).tolist(), TARGET_SR
            )
            text = self._text_from_transcript(transcript)
            n_lines = len(getattr(transcript, "lines", []) or [])
            out.append(Transcription(text=text, meta={"vad_lines": n_lines}))
        return out
