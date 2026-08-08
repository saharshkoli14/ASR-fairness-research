"""NeMo backend (Parakeet TDT via ASRModel, Canary-Qwen via SALM).

Linux/WSL2 only. Canary-Qwen is cast to bf16 after restore: NeMo restores fp32
(~9.7 GB, silently spills over PCIe on 8 GB cards); bf16 is the model card's
intended inference precision. See feasibility_results.json.
"""

import torch

from .base import Transcription


class NeMoTranscriber:
    # batch_size=1 for accuracy runs, same policy as the HF backend (padding must
    # not be able to influence outputs). See EVAL_SPEC changelog 2026-08-07.
    def __init__(self, repo_id: str, revision: str, kind: str = "asr",
                 cast_bf16: bool = False, track_language: bool = False, batch_size: int = 1):
        self.name = repo_id.split("/")[-1]
        self.repo_id = repo_id
        self.revision = revision  # recorded; NeMo from_pretrained pulls the repo's .nemo artifact
        self.kind = kind
        self.track_language = track_language
        self.batch_size = batch_size

        if kind == "salm":
            from nemo.collections.speechlm2.models import SALM

            model = SALM.from_pretrained(repo_id)
            if cast_bf16:
                model = model.to(torch.bfloat16)
        else:
            from nemo.collections.asr.models import ASRModel

            model = ASRModel.from_pretrained(model_name=repo_id)
        self._model = model.to("cuda").eval()

    def transcribe(self, wav_paths: list[str]) -> list[Transcription]:
        if self.kind == "salm":
            return [self._transcribe_salm(p) for p in wav_paths]
        hyps = self._model.transcribe(list(wav_paths), batch_size=self.batch_size)
        out = []
        for h in hyps:
            text = h.text if hasattr(h, "text") else str(h)
            lang = getattr(h, "langs", None) if self.track_language else None
            out.append(Transcription(text=text, detected_language=str(lang) if lang else None))
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
