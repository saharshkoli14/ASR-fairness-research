"""NeMo backend (Parakeet TDT via ASRModel, Canary-Qwen via SALM).

Linux/WSL2 only. Canary-Qwen is cast to bf16 after restore: NeMo restores fp32
(~9.7 GB, silently spills over PCIe on 8 GB cards); bf16 is the model card's
intended inference precision. See feasibility_results.json.
"""

from pathlib import Path

import torch

from .base import Transcription


class NeMoTranscriber:
    # batch_size=1 for accuracy runs, same policy as the HF backend (padding must
    # not be able to influence outputs). See EVAL_SPEC changelog 2026-08-07.
    def __init__(self, repo_id: str, revision: str, kind: str = "asr",
                 cast_bf16: bool = False, track_language: bool = False, batch_size: int = 1,
                 chunk_s: float | None = 120.0):
        self.name = repo_id.split("/")[-1]
        self.repo_id = repo_id
        self.revision = revision  # recorded; NeMo from_pretrained pulls the repo's .nemo artifact
        self.kind = kind
        self.track_language = track_language
        self.batch_size = batch_size
        # Conformer relative-attention memory grows ~O(T^2): EdAcc's 536 s utterance
        # (index 3277) reliably faults the driver on 8 GB, while 199 s succeeds. Chunk
        # above 120 s for margin — affects 16/9177 utterances (0.17%). Recorded per
        # utterance in meta and reported per group (EVAL_SPEC §5).
        self.chunk_s = chunk_s

        if kind == "salm":
            from nemo.collections.speechlm2.models import SALM

            model = SALM.from_pretrained(repo_id)
            if cast_bf16:
                model = model.to(torch.bfloat16)
        else:
            from nemo.collections.asr.models import ASRModel

            model = ASRModel.from_pretrained(model_name=repo_id)
        self._model = model.to("cuda").eval()

    def _split_if_long(self, wav_path: str, tmpdir: str) -> list[str]:
        """Return [wav_path] or a list of <=chunk_s pieces written into tmpdir."""
        import soundfile as sf

        if not self.chunk_s:
            return [wav_path]
        info = sf.info(wav_path)
        if info.duration <= self.chunk_s:
            return [wav_path]
        data, sr = sf.read(wav_path, dtype="float32")
        step = int(self.chunk_s * sr)
        paths = []
        for k, start in enumerate(range(0, len(data), step)):
            p = str(Path(tmpdir) / f"{Path(wav_path).stem}_c{k}.wav")
            sf.write(p, data[start:start + step], sr)
            paths.append(p)
        return paths

    def transcribe(self, wav_paths: list[str]) -> list[Transcription]:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            plan = [(p, self._split_if_long(p, td)) for p in wav_paths]
            flat = [c for _, chunks in plan for c in chunks]

            if self.kind == "salm":
                texts = [self._transcribe_salm(c).text for c in flat]
                langs = [None] * len(flat)
            else:
                hyps = self._model.transcribe(flat, batch_size=self.batch_size)
                texts, langs = [], []
                for h in hyps:
                    texts.append(h.text if hasattr(h, "text") else str(h))
                    lg = getattr(h, "langs", None) if self.track_language else None
                    langs.append(str(lg) if lg else None)

            out, i = [], 0
            for _, chunks in plan:
                n = len(chunks)
                joined = " ".join(t.strip() for t in texts[i:i + n] if t)
                lang = next((l for l in langs[i:i + n] if l), None)
                meta = {"chunked": n} if n > 1 else {}
                out.append(Transcription(text=joined, detected_language=lang, meta=meta))
                i += n
        return out

    def _transcribe_salm(self, wav_path: str) -> Transcription:
        answer_ids = self._model.generate(
            prompts=[[{
                "role": "user",
                "content": f"Transcribe the following: {self._model.audio_locator_tag}",
                "audio": [wav_path],
            }]],
            max_new_tokens=256,
        )
        return Transcription(text=self._model.tokenizer.ids_to_text(answer_ids[0].cpu()))
