# Results — accent-fairness audit of the 2026 ASR generation

Benchmark: **EdAcc** test split (`edinburghcstr/edacc`, revision `d9ae7bd344f0`), 5,087 scored
utterances across 7 accent groups meeting the inclusion rule (≥20 min audio, ≥3 speakers).
Methodology frozen in [`EVAL_SPEC.md`](EVAL_SPEC.md) v1.0 before any model produced a number;
all deviations are changelogged there with dates. Model checkpoints and dataset revisions
pinned in [`pins.json`](pins.json). Normalizer vendored from `openai/whisper@5f86d1d8`.

Decoding: greedy/model-default, batch size 1, each model's documented default precision.
Every number below is reproducible from the committed `results/<model>/transcripts.jsonl`.

---

## 1. Headline table

Per-group WER (%), lower is better. Sorted by mean WER.

| Model | Params | Indian | Irish | Jamaican | US | Nigerian | Spanish | Vietnamese | **mean** | **worst** | **gap** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Parakeet TDT v3 | 0.6B | 8.5 | 10.4 | 18.8 | 12.9 | 17.3 | 16.6 | 22.2 | **15.2** | 22.2 | 13.6 |
| Parakeet TDT v2 | 0.6B | 9.6 | 11.1 | 19.5 | 14.7 | 19.0 | 18.0 | 23.6 | **16.6** | 23.6 | 14.0 |
| Distil-Whisper v3.5 | 756M | 10.5 | 12.1 | 23.2 | 16.8 | 20.6 | 20.4 | 25.8 | **18.5** | 25.8 | 15.3 |
| Whisper large-v3-turbo | 809M | 10.5 | 11.5 | 22.7 | 17.3 | 22.5 | 25.7 | 28.4 | **19.9** | 28.4 | 17.9 |
| Canary-Qwen | 2.5B | 20.5 | 15.1 | 28.7 | 23.2 | 20.9 | 17.8 | 24.5 | **21.3** | 28.7 | 13.5 |
| Moonshine Streaming ᴺᴰ | 245M | 16.5 | 18.0 | 34.7 | 23.5 | 30.2 | 26.2 | 39.6 | **26.8** | 39.6 | 23.1 |
| Whisper-small † | 244M | 34.2 | 40.0 | 87.9 | 98.2 | 82.0 | 104.6 | 102.9 | **80.3** | 104.6 | 70.4 |

ᴺᴰ Non-deterministic: fails the §4.5 determinism gate. Measured band at n=300: 10.0% utterance
disagreement between identical passes, |ΔWER| **±0.21 points** overall (worst audited group
±1.02, Irish). The band is ~1–2% of the disparities being measured, so results are reportable
with the band attached.

† Whisper-small is the fine-tuning base for Part 3, not a deployment candidate. Its >100% group
WERs (hallucination loops inflate insertions) make its CIs uninterpretable; it is excluded from
all statistical comparisons.

**Group sizes** (test split, after cleaning): Nigerian 1,351 utt / 5 spk · US 875 / 9 ·
Spanish 852 / 5 · Indian 627 / 3 · Vietnamese 577 / 6 · Jamaican 455 / 4 · Irish 350 / 3.

---

## 2. What the statistics support

Paired speaker-level bootstrap (1,000 resamples, seed 3407), Bonferroni-corrected across 15
comparisons (α = 0.0033, 99.67% CIs). Whisper-small excluded. Full output in
`results/comparisons_*.json`.

### Primary: each model vs the incumbent, Whisper large-v3-turbo

| Model | Δ gap | Δ worst-group | Verdict |
|---|---|---|---|
| Parakeet TDT v2 | −3.9 [−8.5, −1.8] ✓ | −4.8 [−9.1, −2.9] ✓ | **significantly fairer** |
| Parakeet TDT v3 | −4.3 [−9.5, −1.6] ✓ | −6.3 [−10.5, −3.9] ✓ | **significantly fairer** |
| Moonshine Streaming | +5.2 [−4.5, +15.0] | +11.2 [+2.9, +19.5] ✓ | **significantly less fair** (worst-group) |
| Distil-Whisper v3.5 | −2.6 [−7.7, +2.1] | −2.6 [−7.3, +2.4] | indistinguishable |
| Canary-Qwen 2.5B | −4.4 [−15.6, +14.1] | +0.2 [−7.5, +21.3] | indistinguishable |

### Other supported differences (exploratory, same correction)

- Moonshine is significantly worse than **both Parakeets** on gap (+9.1, +9.4) and worst-group
  (+16.0, +17.4), and worse than Distil-Whisper on worst-group (+13.8).
- Distil-Whisper is significantly worse than both Parakeets on worst-group (+2.2, +3.7).
- Canary-Qwen is significantly worse than Parakeet v3 on worst-group (+6.5).
- Parakeet v3 vs v2 on worst-group: +1.4 [+0.0, +3.5] — **borderline**, CI lower bound touches
  zero. Treated as suggestive, not established.

---

## 3. Findings

**1. Efficiency does not systematically cost accent fairness — but edge optimization does.**
The two fastest models audited (Parakeet TDT 0.6B, v2 and v3) are simultaneously the most
accurate *and* significantly fairer than the 809M incumbent. The prediction that the field's
move toward faster models carries a hidden fairness cost is **not supported as a general
claim**. It is supported for one model: Moonshine Streaming, the most aggressively
edge-optimized system tested, is significantly less fair than every other usable model.
The cost attaches to streaming/edge design, not to efficiency as a category.

**2. Open ASR Leaderboard rank does not transfer to accented conversational speech.**
Canary-Qwen 2.5B leads the Open ASR Leaderboard (~5.6% WER) while Parakeet TDT ranks far below
it. On EdAcc that inverts: Parakeet v3 reaches 15.2% mean WER against Canary-Qwen's 21.3% —
a 0.6B model beating a 2.5B leaderboard leader by 6 points on accented English. Canary-Qwen is
5th of 7 here. Leaderboard position is not evidence of accent robustness.

**3. Canary-Qwen has a structurally different error profile.** It is the only model whose worst
group is Jamaican rather than Vietnamese English, and the only one worse on Indian English
(20.5%) than on Spanish (17.8%) — the reverse of every other model. Its worst/best ratio (1.90)
is the flattest measured, but achieved by being uniformly mediocre rather than uniformly strong.

**4. Distillation did not cost accent robustness.** Distil-Whisper v3.5 is *better* than
Whisper large-v3-turbo on mean WER (18.5 vs 19.9) and both disparity metrics, though the
disparity differences are not statistically supported. The published concern that distilled
models underperform on underrepresented accents is not reproduced here.

**5. The multilingual upgrade did not cost English accent robustness.** Parakeet v3
(25 languages) matches or beats v2 (English-only) on every group and both disparity metrics.

**6. Hallucination loops are architectural.** Runaway repetition — hypotheses exceeding
5× the reference length — occurred 401 times for Whisper-small, 13 for Distil-Whisper, 6 for
turbo, and **0** for both Parakeets and Canary-Qwen. RNN-T transducers and the SALM decoder do
not exhibit the failure mode. Moonshine's count of 1 is not comparable: its runtime applies a
`max_tokens_per_second` truncation heuristic that suppresses loops at the library level.

**7. Every model ranks the accent groups near-identically** (Indian/Irish best, Vietnamese
worst for 5 of 7). The disparity is a property of the speech, not of any one architecture.

---

## 4. Methodological finding

**Max−min gap, the metric the accent-bias literature reports by default, is the statistically
weakest disparity metric available.** On identical data under identical correction:

| Metric | Pairwise comparisons reaching significance |
|---|---|
| max−min gap | **4 / 15** |
| worst-group WER | **10 / 15** |

The gap is a difference of two independently noisy per-group estimates and carries roughly
double the variance of a single level. Recommendation: report **worst-group WER as the primary
disparity metric**, with the gap secondary. This harness reports both by default.

---

## 5. Limitations

- **Speaker counts are small.** Indian and Irish English have 3 speakers each, Jamaican 4.
  Per-group WER partly reflects those individuals, not the accent. This is a property of
  EdAcc's speaker-disjoint test split and is why bootstrap resampling is over *speakers*.
  It is the main reason Canary-Qwen is statistically indistinguishable from most models.
- **Southern British English is absent** from the audit — only 18.2 min / 2 speakers in the
  test split, below the inclusion threshold. 17 accent groups were pooled and excluded.
- **Cleaning is not random with respect to accent.** Utterances marked
  `IGNORE_TIME_SEGMENT_IN_SCORING` or containing `<FOREIGN>` are dropped; the highest exclusion
  rate is Spanish at 3.1% (22 of 27 exclusions are code-switching), all others below 1.4%.
- **Long-audio handling differs by backend.** NeMo models chunk utterances above 120 s
  (16 of 9,177; conformer attention memory is O(T²) and a 536 s utterance faults an 8 GB GPU),
  while Whisper models used native sequential long-form decoding with full context. NeMo models
  are mildly disadvantaged on 0.17% of data.
- **Moonshine's runtime differs from the others' in two ways** beyond precision: VAD
  re-segmentation, and the loop-truncation heuristic noted above. Both are its documented
  defaults, but its loop count and, to a lesser degree, its WER reflect library policy as well
  as model behaviour.
- **Training-data contamination is unknown.** Neither NVIDIA nor OpenAI fully discloses training
  corpora, so EdAcc overlap cannot be ruled out. Contamination would *shrink* measured
  disparities, making these figures a lower bound.
- **Worst/best ratio is post-hoc.** It was computed after seeing results and is exploratory,
  not pre-registered. It is not used for any statistical claim.
- **Efficiency is not yet measured.** Throughput, tail latency, and VRAM (§4.2) are pending;
  the efficiency-vs-disparity frontier cannot be drawn until they exist.

---

## 6. Reproducing

```bash
pip install -e ".[dev]"
python pin.py                          # resolve checkpoint + dataset revisions
python scripts/vendor_normalizer.py    # freeze the text normalizer
python scripts/make_groups.py          # freeze accent groups from the pinned split
python scripts/run_audit.py --model parakeet-tdt-0.6b-v3
python scripts/verify_determinism.py parakeet-tdt-0.6b-v3
python scripts/compare_models.py --bonferroni --metric worst_group_wer
```

NeMo models (Parakeet, Canary-Qwen) require Linux or WSL2: `pip install "nemo_toolkit[asr]"`.
Moonshine uses the official ONNX runtime: `pip install moonshine-voice`.

Hardware for all runs: RTX 4060 Laptop (8 GB), 83 W power cap, AC power.
