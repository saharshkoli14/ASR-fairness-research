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
- **Service model (added 2026-08-11)**: one model instance, serving one request at a time —
  every backend serializes its model call on a per-instance lock. Concurrency is offered load,
  not parallel execution, so the reported figures are queueing latencies. This is what a single
  GPU actually provides, and holding it identical across backends is what makes the numbers
  comparable between models. A run in which any request raises is **discarded, not reported**:
  a sweep that lost workers measures an unknown offered load.
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
- **Multiplicity (added 2026-08-10, before any comparison was published).** All-pairs testing
  over 7 models is 21 comparisons per metric; at α=0.05 roughly one false positive per metric
  is expected. Therefore:
  1. **Primary comparisons** are pre-specified and Bonferroni-corrected within their family:
     (a) each efficiency-generation model vs `whisper-large-v3-turbo` as the incumbent baseline,
     (b) `parakeet-tdt-0.6b-v2` vs `v3` (the multilingual-upgrade question).
  2. All other pairs are **exploratory**, reported with uncorrected CIs and labelled as such.
  3. `whisper-small` is excluded from the comparison family — it is the fine-tuning base, not a
     deployed candidate, and its >100% group WERs make its CIs uninterpretable.
  Run corrected: `python scripts/compare_models.py --bonferroni`.

## 4.5 Determinism gate (added 2026-08-08 — required before any model's numbers are reported)

A model's results are only reportable if its outputs are **position-independent**: transcribing
the same audio must give the same text regardless of how many prior calls the process has made.

Check: sample ≥ 25 utterances spread across a completed run, re-transcribe in a fresh process,
diff against the stored hypotheses (`scripts/verify_determinism.py <model>`). Any mismatch means
the run is not reproducible and the model is excluded until a deterministic runtime is found.

Rationale: discovered empirically — `UsefulSensors/moonshine-streaming-medium` produces
*different transcriptions for identical audio* (3/20 mismatches under `transformers`, 1/25
under the official ONNX runtime), while all Whisper-family models pass 25/25. Without this
gate, a non-reproducible model's WER would have entered the results table indistinguishably
from valid numbers.

**Failing the gate is not automatic exclusion (amended 2026-08-08).** A model that fails must
have its instability *quantified* before any number is reported:

1. Re-transcribe a fixed ≥300-utterance subset a second time under identical configuration.
2. Report: utterance-level disagreement rate, and |ΔWER| between the two passes, per group
   and overall (`scripts/measure_nondeterminism.py`).
3. The model is reportable only if the band is small relative to the effect being measured
   (the accent gap). Every headline number for that model carries the band; the results table
   marks it non-deterministic.
4. If the band is comparable to the gap, the model is excluded — its numbers cannot support
   the claim.

Moonshine diagnosis: instability persists with a freshly constructed transcriber and freshly
loaded weights, and the differing utterance flips between the same two near-equal variants.
This is run-to-run floating-point nondeterminism (ONNX Runtime reduction order / thread
scheduling), not accumulated session state — so process isolation cannot fix it and a measured
band is the correct treatment.

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
- 2026-08-12: **DRO weights made scale-invariant; τ sweep set to {1.0, 0.5, 0.25} (§6).** The
  stationary softmax above used *absolute* per-group loss, which self-annihilates as training
  converges: every group's loss shrinks toward zero, so the differences between them shrink too
  and q drifts to uniform. Measured on the τ=0.3 run — weight ratio 2.79× at step 700, 1.59× at
  2800, **1.18× by 6300** — i.e. from roughly step 3000 the DRO arm *was* ERM, and every snapshot
  (4200+) sat in that regime, while the *relative* gap it was supposed to act on was undiminished
  (hausa 0.16 vs zulu 0.08, a 2× ratio). Left uncorrected, all three arms of the sweep would have
  been ERM in disguise and the null result would have been an artefact of the weighting rule.
  Fix: normalise by the current mean loss before the softmax, `q_g ∝ exp((L̄_g / L̄) / τ)`, so the
  tilt depends on how much *harder* a group is rather than on absolute loss scale — at step 5600
  that yields 7.9× at τ=0.25 against the 1.4× actually obtained. Sweep re-calibrated to
  **τ ∈ {1.0, 0.5, 0.25}** for weak/medium/strong tilt under the new parameterisation. Run
  `dro_tau0.3` discarded at step ~6800; no numbers from it are reported.
- 2026-08-12: **DRO group weights changed from cumulative to stationary; η sweep replaced by a
  temperature sweep (§6 deviation).** §6 specifies online exponentiated gradient,
  `q_g ← q_g · exp(η·L_g)`, with η ∈ {0.01, 0.1, 1.0}. That rule weights by the *running product*
  of past losses, so its concentration grows without bound in the training horizon. Measured, not
  assumed: at η=0.01 — the **smallest** value specified — an EMA loss spread of only 0.38 nats
  drove **q(hausa) = 0.87 at step 1450 of 16,800**, every other group at the 1e-3 floor, with all
  four snapshots post-collapse. Larger η collapses sooner, so the sweep as written could only ever
  have produced single-group models, and the ERM-vs-DRO comparison would not have existed.
  Replacement: `q_g ∝ exp(L̄_g / τ)` over the current per-group EMA loss — stationary, bounded by
  construction, and self-correcting (a group's weight falls as soon as the model improves on it,
  which the cumulative form cannot do). Still exactly one extra hyperparameter, as §6 requires;
  swept over **τ ∈ {1.0, 0.3, 0.1}**, giving max/min weight ratios of 1.5× / 3.5× / 44× at the
  observed loss spread. Selection remains validation worst-group WER, budgets remain identical to
  ERM's 16,800 steps. The collapse itself is retained as a Part 3 finding: both failure modes —
  chasing group *frequency* under natural sampling, then chasing cumulative *history* — are
  invisible in the Group-DRO literature's usual group-balanced setting.
- 2026-08-12: **Group-DRO update corrected before the sweep (§6).** The textbook online update
  `q_g <- q_g * exp(eta * L_g)` touches only the groups present in the batch. That is sound under
  group-balanced sampling, but this corpus is sampled naturally and is imbalanced 11.5×: Yoruba
  (46% of utterances) appears in nearly every batch and collects a multiplicative boost each step,
  while Zulu (4%) is shrunk by renormalisation whenever it is absent. A 300-step probe collapsed
  to **q(yoruba)=0.87 with Zulu pinned at the floor** — DRO chasing the *largest* group rather
  than the hardest, i.e. the objective inverted. Fix: maintain an EMA (β=0.95) of per-group loss
  and reweight **every observed group each step** from it, decoupling update frequency from group
  frequency. Rejected alternative: group-balanced sampling, which would make the sampler a second
  difference between arms and confound the ERM/DRO contrast; the EMA leaves the loaders identical
  so the arms differ only in loss reduction. Also added: renormalisation of weights over the
  groups present in a batch (keeps DRO on ERM's loss scale, so the shared LR schedule means the
  same thing in both arms) and a floor q ≥ 1e-3 (the multiplicative update is otherwise absorbing
  at zero — a starved group can never recover however badly the model does on it). Post-fix probe:
  q tracks per-group loss, with hausa — the group ERM leaves worst at 24.4% test WER — carrying
  the highest weight. Known property, not a defect: EG weights by *cumulative* loss, so
  concentration grows with the training horizon; the η sweep {0.01, 0.1, 1.0} is what tests it.
- 2026-08-12: **Checkpoint-selection tiebreak added (§6).** The ERM sweep's top two checkpoints
  differed by **0.0009 WER points** on validation (step4200 20.9201, step12600 20.9210, n=1,346) —
  the selection criterion does not separate them, and the winner was being decided by filesystem
  ordering. Rule, stated now rather than after seeing test numbers: within **0.05 WER points**,
  prefer the checkpoint with **fewest training steps** (cheaper, and less memorised — training
  loss kept falling for 6 epochs after validation WER flattened). This is not cosmetic: the tied
  checkpoints disagree on the disparity figures ERM will be compared against (worst-group 25.08
  vs 24.85, gap 6.03 vs 5.81), so `selected.json` records the tied set and their disparity
  metrics as an explicit sensitivity note. Both arms use the identical rule.
  **Budget note:** §6's "identical budgets" governs *training*, not selection — the DRO arm
  therefore trains the full 16,800 steps with the same snapshot grid, and selects from it by
  validation worst-group WER. Shortening DRO's training because ERM selected an early checkpoint
  would break the comparison.
- 2026-08-12: **Training-time filters on the AfriSpeech subset (§6).** Two model-imposed limits,
  applied identically to both arms and enforced at dataset construction so a violating sample
  fails at startup rather than mid-run: (a) **labels > 448 tokens** — Whisper's
  `max_target_positions`, a hard limit that raised `ValueError` 150 steps into the first ERM
  attempt; (b) **audio > 30 s** — Whisper's encoder window, so the reference for a longer
  utterance describes audio the model never sees, and training on those pairs teaches the decoder
  to invent the tail. Effect: 173 of 8,400 train utterances dropped (2.1%), 24.62 h → 22.00 h,
  entirely from the audio-length rule (0 exceeded the token limit after it). By group: hausa 65,
  yoruba 56, igbo 44, swahili 8, **zulu 0** — the smallest group is untouched, so the 11.5×
  imbalance under test is preserved. Recorded per run in `run_config.json` under `filters`.
- 2026-08-11: **Part 3 data decisions frozen (§6), before any training run.** AfriSpeech-200 groups
  frozen in `groups_afrispeech.json` by applying §3's rule (≥20 min, ≥3 speakers) to train **and**
  test independently: 38 accents qualify in train, 11 in test, **5 in both** (hausa, igbo, swahili,
  yoruba, zulu; 102.2 h train, 59% of the split). Test is the binding constraint. Two amendments:
  1. **Train subsampled to ~25 h, proportionally** (seed 3407, speaker-stratified), preserving the
     11.7× Yoruba:Zulu imbalance. Balancing the groups would delete the phenomenon Group-DRO is
     being tested on; the subsample keeps it while making a 4-run sweep feasible on an 8 GB laptop
     (~3–5 h/run vs 12–20 h at full size). Both arms use the identical subsample — §6 requires
     equal budgets, and a budget difference would confound the ERM/DRO comparison.
  2. **Validation augmented for thin groups.** Hausa's dev split is 15.4 min, below the threshold
     its own group was selected by, and §6 selects the DRO arm on validation *worst-group* WER —
     i.e. precisely where the estimate is weakest. Any group with <20 min dev receives a
     **speaker-disjoint** holdout from train until it reaches 20 min (only hausa qualifies); held-out
     speakers are excluded from training. Alternative considered and rejected: selecting DRO by mean
     WER, which §6 already warns sandbags DRO.
  Audio is resampled 44.1 kHz → 16 kHz mono once and cached (Whisper's required rate; caching avoids
  re-resampling every epoch and keeps the working set ~3 GB against 32 GB of source tarballs).
- 2026-08-11: **Cross-environment scoring verified; the headline table is internally comparable.**
  Per-run provenance revealed that the audit spans two environments carrying **different major
  versions of jiwer** — 3.1.0 under WSL2 (NeMo models) and 4.0.0 on native Windows (HF models) —
  and `summary.json` records no library versions, so which version produced which row of the
  headline table is unrecoverable after the fact. If the two disagreed at all, the per-group WERs
  would not be mutually comparable and every disparity metric built on them would be unsafe.
  Tested rather than assumed: `scripts/verify_scoring_consistency.py` re-scores all 7 models from
  the committed `transcripts.jsonl` in a single interpreter (jiwer 4.0.0), replaying the scoring
  path verbatim. **All 7 reproduce exactly — max |Δ| = 0.00e+00 across every per-group WER,
  micro/macro, worst-group, gap, std, speaker and word count**, 5,087 utterances each. The jiwer
  split is therefore harmless, and RESULTS.md's reproducibility claim is now checked, not asserted.
  The underlying gap (accuracy runs record no library versions) remains open.
- 2026-08-11: **Throttling verdicts made three-valued; thermal protocol scoped to GPU runs (§4.2).**
  The >10% clock-drop check fired twice for reasons unrelated to heat. (a) On the CPU-only
  Moonshine run the sampler watched an *idle* GPU (5–13 W, 210 MHz floor): a transient boost
  falling back to idle read as a 42.8% "drop". GPU thermal records are now voided when the
  execution device is not CUDA, and the run is marked thermal-steady-state-unverified — no CPU
  thermal sampling exists yet, which is a real gap for the one CPU model. (b) Parakeet is fast
  enough that a batch phase lasts 10–18 s, giving ~10 samples, so the start/end means are two
  samples each and DVFS jitter dominates — one phase reported a −98.7% "drop" (the clock rose).
  Verdicts now require ≥30 samples and are otherwise `null` + `throttle_verdict: indeterminate`,
  distinct from `false`. Applied to the existing runs: no model shows measured throttling;
  Parakeet v2/v3 and Moonshine are indeterminate on every phase. Supporting evidence that the
  Parakeet runs were nonetheless at steady state: 71–80 °C with sustained power at the 83 W cap
  and clocks in the same 2.2–2.5 GHz band as the long, well-sampled Whisper runs. Only the
  derived verdict was recomputed; sampled clocks, temperatures and power are unchanged.
  Protocol amendment for any future rerun: extend timing phases to ≥30 s so fast models get a
  real verdict rather than an absent one.
- 2026-08-11: **Per-run provenance added to efficiency runs (§5, §7).** `efficiency.json` recorded
  only GPU, platform and Python version — no harness commit (§7) and no library/CUDA versions
  (§5). `pins.json`'s `env_at_pin_time` does not substitute: it describes one machine at pin time
  (Windows, py3.14, torch 2.11+cu128), while the NeMo and Whisper runs execute under WSL2 on a
  different interpreter and torch build, and Moonshine runs on native Windows CPU. New module
  `asr_fairness_audit.provenance` records commit + dirty flag + installed versions per run.
  Runs predating this are stamped by `scripts/stamp_efficiency_provenance.py`, marked
  `backfilled: true` — versions read post-hoc from an unchanged environment are weaker evidence
  than values written by the run, and the flag preserves that distinction. Known remaining gap:
  `summary.json` (accuracy runs) carries `harness_commit` but still no library versions.
- 2026-08-11: **Efficiency latency runs discarded; service model made explicit (§4.2).** The
  runtimes are not thread-safe, and the closed-loop harness let a raising request kill its
  worker. Both NeMo backends therefore lost every worker past the first, on its first request:
  RNNT `transcribe()` freezes the encoder on entry and calls `unfreeze(partial=True)` on exit,
  so an interleaved second exit raises `ValueError: Cannot unfreeze partially…`; SALM
  `generate()` detaches `llm.model.embed_tokens` to splice in audio embeddings, so a concurrent
  caller raises `AttributeError: 'Qwen3Model' object has no attribute 'embed_tokens'`. The
  written-out concurrency_2 / concurrency_4 figures for parakeet-tdt-0.6b-v2, -v3 and
  canary-qwen-2.5b are single-worker numbers with 1 and 3 requests silently dropped (n=119 and
  n=117 of 120) — invalid, not merely noisy. Fixes: (a) all backends serialize model calls on a
  per-instance lock (`backends.base.SerializedInference`); (b) `closed_loop` keeps a worker
  alive after a failed request, counts it, and raises rather than emitting a partial sweep.
  Consequence: **all 7 efficiency runs are rerun.** The HF/Moonshine runs did not crash, but
  they were measured with overlapping pipeline calls and so describe a different service model
  than the reruns; keeping them would compare models under two different definitions of
  concurrency. Existing `results/*/efficiency.json` are flagged `invalidated` in place. No
  reported number depended on them (RESULTS.md §Efficiency was still "not yet measured").
- 2026-08-10: **Determinism gate results — all 7 models assessed.** Deterministic (25/25 identical):
  whisper-large-v3-turbo, distil-large-v3.5, whisper-small, parakeet-tdt-0.6b-v2,
  parakeet-tdt-0.6b-v3, canary-qwen-2.5b. Failing: moonshine-streaming-medium, band measured
  per §4.5 at n=300 — 10.0% utterance disagreement, |ΔWER| 0.21 points overall, worst audited
  group 1.02 points (Irish English). Band is ~1–2% of the accent gaps being measured (13–23
  points), so Moonshine is REPORTABLE with the band attached to every headline number and a
  non-deterministic marker in the results table. Note: the n=300 run predates restricting the
  sample to audited groups; re-run before writeup so the published band excludes pooled groups.
- 2026-08-10: **NeMo long-audio chunking at 120 s.** EdAcc test index 3277 is 536 s; Conformer
  relative-attention memory grows ~O(T²) and this utterance reliably faults the GPU driver on
  8 GB (199 s succeeds, 536 s does not — reproduced twice at the identical index). Utterances
  over 120 s are split, transcribed independently, and joined; 16/9177 (0.17%) are affected.
  Recorded per utterance (`meta.chunked`) and reported per group in each summary, because long
  turns cluster by speaker and therefore by accent. Comparability note: Whisper-family models
  processed these same utterances with their native sequential long-form decoding (full
  context), so NeMo models are mildly disadvantaged on the affected 0.17%.
- 2026-08-08: **Added §4.5 determinism gate.** Moonshine Streaming (transformers) fails it:
  identical audio yields different transcriptions depending on process state (3/20 sampled
  utterances differed between mid-run and fresh-process transcription), and separately triggers
  a CUDA index assert after several hundred sequential calls. Isolation work ruled out audio
  duration (0.05 s–60 s all pass), per-utterance content (all 16 crash-region utterances pass
  individually), and attention backend (SDPA and eager both affected). Whisper-family models
  pass the gate 25/25 each (turbo, distil-large-v3.5, whisper-small) — their results stand.
  Moonshine transcripts produced before this date are discarded as contaminated.
- 2026-08-08: Corrected HF-backend dtype to model-card defaults per §5 (was uniformly bf16 —
  implementation error): Whisper family fp16, Moonshine fp32. A/B on 60 loop-flagged
  whisper-small utterances: 57/60 loops persist at fp16 — loop behavior is real, not
  precision-induced. Whisper-family runs regenerated at fp16 for config compliance; bf16
  transcripts retained in git history. Moonshine addendum: inputs padded to 80-sample frame
  multiple (model requirement).
- 2026-08-07: **Accuracy runs use batch_size=1** (decided before any full run completed).
  Empirical: batched padding changed whisper-small outputs on 6/50 smoke utterances, and
  utterances > 30 s crash batched Whisper padding while long-form decoding engages only at
  batch 1. Efficiency runs (§4.2) may still batch — throughput at batch is part of what they
  measure; output text from efficiency runs is never scored.
- 2026-08-07: Added secondary diagnostic (before any full audit run; no reportable numbers
  existed): per-group hallucination-loop rate, flagged when normalized hypothesis exceeds
  max(10, 5× reference words). Motivated by smoke-run observation: whisper-small produced a
  356-word repetition loop on a 2-word utterance, reproduced at batch=1 (real model behavior,
  not a batching artifact). Loops REMAIN in WER (deployed-default behavior); the diagnostic
  makes their frequency and group correlation visible. Loop rate differing by accent group
  would itself be a fairness finding.
- 2026-08-06: Feasibility complete — all 7 checkpoints load and transcribe on RTX 4060 (WSL2 for
  NeMo models). Resolved decision 4 (concurrency 1/2/4). Findings recorded for writeup: NeMo
  restores Canary-Qwen in fp32 (~9.7 GB → silent PCIe spill on 8 GB; bf16 cast → 4.86 GB, 3×
  faster); Moonshine Streaming works with SDPA (no flash-attn needed). Canary-Qwen inference
  protocol addendum: cast to bf16 after restore — this IS the documented intended precision (§5).
  **Spec frozen v1.0.**
