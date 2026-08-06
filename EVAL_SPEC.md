# Evaluation Specification — ASR Accent-Fairness Audit

Status: **FROZEN v1.0 (2026-08-06)** — all four opening decisions resolved. Any later change
requires a dated changelog entry at the bottom, made *before* the affected numbers are produced.

Principle: every choice here is locked **before** any model produces a number that goes in the paper.
No metric, grouping, or normalization changes after seeing results.

---

## 1. Models under test

| Model | HF checkpoint | Role | Revision (full SHA in `pins.json`) |
|---|---|---|---|
| Canary-Qwen 2.5B | `nvidia/canary-qwen-2.5b` | Accuracy frontier | `b1469e1bba1c` |
| Parakeet TDT 0.6B v2 | `nvidia/parakeet-tdt-0.6b-v2` | Speed frontier (English-specialized) | `ae9ad07059c7` |
| Parakeet TDT 0.6B v3 | `nvidia/parakeet-tdt-0.6b-v3` | Speed frontier (multilingual successor) | `541d1f99c6b0` |
| Whisper large-v3-turbo | `openai/whisper-large-v3-turbo` | Pruned-decoder baseline | `41f01f3fe87f` |
| Distil-Whisper | `distil-whisper/distil-large-v3.5` | Distillation probe | `728a7691f3ff` |
| Moonshine Voice | `UsefulSensors/moonshine-streaming-medium` | Edge/streaming (2026 gen) | `57b843633a8c` |
| Whisper-small | `openai/whisper-small` | Fine-tuning base (ERM + Group-DRO) | `973afd24965f` |

Datasets: EdAcc `d9ae7bd344f0` (matches the revision inspected for §4.3 marker rules),
AfriSpeech-200 `b538c6e11191`. Pinned 2026-08-06 via `pin.py`; full SHAs in `pins.json`.

**Resolved 2026-08-06:** Parakeet — both v2 and v3 (v2→v3 English accent-robustness delta is a
free finding). Moonshine — Voice family, `moonshine-streaming-medium` (245M), in `transformers`
as "Moonshine Streaming" since Feb 2026. *Caveat:* its sliding-window encoder prefers
flash-attention, which is painful on Windows; verify SDPA fallback during feasibility check
before freeze.

Pinning: resolve each repo's `main` commit SHA at spec-freeze time via
`huggingface_hub.HfApi().model_info(repo_id).sha`, record it in the table above, and pass
`revision=` in every load call. Same for datasets. A `pin.py` script does this; its output is
committed.

## 2. Datasets

### 2.1 Primary: EdAcc — `edinburghcstr/edacc`
- License CC-BY-SA. Splits: `validation` (9,848 utts), `test` (9,289 utts).
- Fields: `speaker`, `text`, `accent` (linguist-annotated), `raw_accent`, `gender`, `l1`, `audio`.
- **All headline numbers come from `test`.** `validation` is for harness debugging and
  fine-tuning model selection only.
- Revision: `d9ae7bd344f0` (full SHA in `pins.json`).

### 2.2 Secondary: AfriSpeech-200 — `intronhealth/afrispeech-200`
- Used for the fine-tuning experiment (train split) and as a second audit test set.
- Speaker-disjoint splits already enforced by the dataset. Not all accents have all splits —
  group inclusion rule (§3) applies per split.
- Revision: `b538c6e11191` (full SHA in `pins.json`).

### 2.3 Excluded (and why — goes in LIMITATIONS.md)
- **Common Voice**: self-reported accent labels, noisy; would contaminate group definitions.
- **L2-ARCTIC**: read speech, 24 speakers — too small for per-group WER with CIs. May be used
  qualitatively in the demo Space only.

### 2.4 Contamination
Whether EdAcc/AfriSpeech test audio appears in any model's training data is **unknown and mostly
unknowable** (NVIDIA and OpenAI do not fully disclose training corpora). Mitigation: state it in
LIMITATIONS.md; note that contamination would *shrink* measured disparities, making the audit
conservative — a found gap is a lower bound.

## 3. Accent groups

- Grouping variable: EdAcc `accent` (linguist-annotated). Never `raw_accent`, never self-report.
- **Inclusion rule:** a group enters the audit iff it has ≥ 20 minutes of test audio **and**
  ≥ 3 distinct speakers. Groups below threshold are pooled into `other` and reported in the
  appendix but excluded from disparity metrics (a 1-speaker "group" measures the speaker, not
  the accent).
- The final group list is computed once from the pinned dataset revision by
  `make_groups.py`, written to `groups.json`, and committed. It is input to the harness,
  never recomputed at eval time.
- AfriSpeech: same rule applied to its `accent` column.

## 4. Metrics

### 4.1 Accuracy
All WER after text normalization (§4.3).

- **Per-group WER**: aggregate errors/words *within* group (micro within group).
- **Mean WER — report both**:
  - *micro*: total errors / total words over the whole test set (comparable to Open ASR Leaderboard);
  - *macro*: unweighted mean of per-group WERs (the fairness-relevant average).
  These diverge when group sizes are skewed; reporting only micro hides small groups. Headline = both.
- **Worst-group WER** (the Group-DRO target).
- **Gap**: max − min per-group WER.
- **Std** across per-group WERs.
- **Uncertainty**: 95% CI on every per-group WER and on the gap, via bootstrap over *speakers*
  (not utterances — utterances within a speaker are correlated), 1,000 resamples, seed 3407.

### 4.2 Efficiency (all on the same hardware, RTX 4060 Laptop 8 GB, specs recorded)
- **RTFx**: audio-seconds transcribed per wall-clock second, batch 1 *and* max batch that fits.
- **Latency under load**: closed-loop client at concurrency 1, 2, 4; report TTFT (streaming models
  only) and end-to-end latency at p50/p95/p99. No zero-load averages anywhere in the results.
- **Peak VRAM**: `torch.cuda.max_memory_allocated()` + `nvidia-smi` sampled at 1 Hz; report max.
- **Thermal protocol**: 5-min warmup before timing; report GPU clock at start and end of each
  timing run; if clock drops > 10%, note throttling in the run log. Timing runs happen with the
  laptop plugged in, performance power profile, ambient conditions logged.
- Efficiency runs are **separate** from accuracy runs (accuracy batching can be whatever is fastest).

### 4.3 Text normalization
- **Whisper `EnglishTextNormalizer`** (from `openai-whisper` / `transformers`), applied to both
  reference and hypothesis — the Open ASR Leaderboard standard, so mean WER is externally comparable.
- Version-pin the normalizer source; vendor the file into the harness so a library upgrade can't
  silently change numbers.
- Sanity check in CI: a fixed list of 20 (ref, hyp) pairs with known normalized WER must
  reproduce exactly.
- **EdAcc-specific rules** (from inspection of 50 raw validation transcripts, 2026-08-06;
  observed dataset revision `d9ae7bd344f0562b766ec93ee5ce8f2f9568ce66`):
  1. **Drop** utterances whose text is exactly `IGNORE_TIME_SEGMENT_IN_SCORING` (corpus-marked
     unscoreable segments).
  2. **Drop** utterances containing `<FOREIGN>`: the tag stands in for spoken non-English content
     absent from the reference — stripping it would charge models insertion errors for correctly
     transcribing real speech. Unscoreable, not strippable.
  3. **Strip** non-speech event tags `<OVERLAP>`, `<LAUGH>`, `<DTMF>` (pattern `<[A-Z_]+>`) from
     references before normalization.
  4. Doubled quotes (`''…''`) and casing are handled by the Whisper normalizer; no extra rule.
  5. **Report per-group exclusion rates** from rules 1–2 in the appendix. Code-switching
     correlates with accent group, so exclusions are non-random; if any group loses > 10% of its
     audio, flag it in LIMITATIONS.md. The cleaning itself must not silently bias the audit.

### 4.4 Statistical comparisons
- Model A vs model B on gap/worst-group: paired bootstrap over speakers, report CI of the
  difference. No claim of "X is less fair than Y" without a CI excluding zero.

## 5. Inference protocol

- Greedy decoding everywhere (no beam search) unless a model's documented default differs —
  then use the documented default and record it. Rationale: audit deployed-default behavior.
- Language forced to English where the API allows (`language="en"`); for auto-detecting models
  (Parakeet v3), record detected language per utterance and count misdetections — an accent
  causing wrong language detection **is itself a fairness finding**, not noise to be cleaned.
- Long audio: EdAcc utterances are pre-segmented; use each model's default chunking for anything
  over its window. Record chunking config per model.
- Audio: resample to each model's expected rate with the model's own processor; no other
  preprocessing.
- Precision: bf16/fp16 per model card default; record per model. Seeds fixed (3407) — note
  greedy ASR is deterministic up to kernel nondeterminism; record `torch`, `nemo`, `transformers`,
  CUDA versions in every results file.

## 6. Fine-tuning experiment (spec'd now so the eval can't drift toward it)

- Base: `openai/whisper-small`, full fine-tune (known PEFT/LoRA incompatibility with Whisper
  encoder — do not burn a weekend rediscovering this; full FT of 244M fits in 8 GB with
  gradient checkpointing + 8-bit Adam, else Kaggle T4).
- Data: AfriSpeech-200 train. Groups: `groups.json` for AfriSpeech.
- Arms: (a) ERM, (b) Group-DRO, identical budgets/schedules; DRO group-weights step size is the
  only extra hyperparameter, swept over {0.01, 0.1, 1.0} on validation worst-group WER.
- Model selection: ERM arm by validation *mean* WER; DRO arm by validation *worst-group* WER.
  (Selecting both by mean would sandbag DRO.)
- Report mean AND worst-group for both arms — DRO degrading mean is an expected, reportable outcome.
- Evaluate both arms on EdAcc test too (cross-corpus generalization of the fairness fix).

## 7. Reproducibility artifacts
- `pins.json`: model SHAs, dataset SHAs, library versions, CUDA version, driver, GPU name.
- `groups.json`: frozen group definitions with per-group speaker/minute counts.
- Every results CSV carries the git commit of the harness that produced it.

---

## Open decisions (blocking freeze)
1. ~~Parakeet v2, v3, or both.~~ **Resolved: both** (2026-08-06).
2. ~~Moonshine Voice vs moonshine-base.~~ **Resolved: Voice / `moonshine-streaming-medium`** (2026-08-06).
3. ~~EdAcc transcript disfluency handling.~~ **Resolved: rules 1–5 in §4.3** (2026-08-06).
4. ~~Concurrency levels for load test.~~ **Resolved: 1/2/4 confirmed** (2026-08-06). Feasibility
   measured peak 4.86 GB reserved for the largest model (Canary-Qwen, bf16) — ~2.6 GB headroom on
   the 7.5 GB budget with a single model instance serving queued requests.

## Changelog
- 2026-08-06: v0.1 initial draft.
- 2026-08-06: Resolved decisions 1–2 (Parakeet both versions; Moonshine Voice medium). Flagged
  flash-attention/Windows risk for Moonshine Streaming.
- 2026-08-06: Resolved decision 3 (EdAcc marker rules) from raw-data inspection. Spec is frozen
  for accuracy methodology; only decision 4 (load-test concurrency) remains, pending feasibility
  measurements.
- 2026-08-06: Feasibility complete — all 7 checkpoints load and transcribe on RTX 4060 (WSL2 for
  NeMo models). Resolved decision 4 (concurrency 1/2/4). Findings recorded for writeup: NeMo
  restores Canary-Qwen in fp32 (~9.7 GB → silent PCIe spill on 8 GB; bf16 cast → 4.86 GB, 3×
  faster); Moonshine Streaming works with SDPA (no flash-attn needed). Canary-Qwen inference
  protocol addendum: cast to bf16 after restore — this IS the documented intended precision (§5).
  **Spec frozen v1.0.**
