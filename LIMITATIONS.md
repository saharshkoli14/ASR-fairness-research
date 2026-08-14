# Limitations

What these results cannot support, collected in one place. EVAL_SPEC.md §4.3 requires
group-level exclusion effects to be reported here; the rest is included because a
reader deciding whether to rely on a number needs to know where it is thin.

Nothing below was discovered after the fact and quietly accommodated — each item is
either pre-registered in EVAL_SPEC.md or carries a dated changelog entry there.

---

## 1. Statistical power

**EdAcc speaker counts are small.** Indian and Irish English have 3 speakers each,
Jamaican 4. Per-group WER partly reflects those individuals rather than the accent.
This is a property of EdAcc's speaker-disjoint test split, and it is why bootstrap
resampling is over *speakers* rather than utterances — resampling utterances would
treat correlated data as independent and produce intervals that are too narrow. It is
the main reason Canary-Qwen is statistically indistinguishable from most models
despite a visibly different error profile.

**The max−min gap is the weakest metric reported.** It is a difference of two
independently noisy per-group estimates and carries roughly twice the variance of a
single level. On identical data under identical correction it reached significance in
4/15 pairwise comparisons against worst-group WER's 10/15 (RESULTS.md §5). Reported
because the accent-bias literature reports it, not because it is the better statistic.

**Part 3's null is bounded, not absolute.** The paired comparison of Group-DRO against
ERM gives Δ worst-group −0.43 points with a 99.67% CI of [−1.69, +3.48]. That excludes
effects larger than roughly 2 points; it does not exclude smaller ones. The defensible
claim is "no detectable effect at this sample size", not "Group-DRO does not work".

---

## 2. Coverage

**Southern British English is absent from the audit.** Only 18.2 minutes across 2
speakers in the EdAcc test split, below the §3 inclusion threshold. 17 accent groups
were pooled and excluded on the same rule.

**AfriSpeech's test split is the binding constraint on Part 3.** 38 accents clear the
§3 rule in train but only 11 in test, leaving 5 groups audited (hausa, igbo, swahili,
yoruba, zulu). Group-DRO was therefore tested over 5 groups, not the ~120 accents the
corpus contains.

**Two corpora, both read or conversational English.** Nothing here speaks to
spontaneous telephony, children's speech, or code-switched speech beyond the exclusions
noted below.

---

## 3. Data cleaning is not random with respect to accent

**EdAcc.** Utterances marked `IGNORE_TIME_SEGMENT_IN_SCORING`, or containing
`<FOREIGN>`, are dropped (§4.3 rules 1–2). Code-switching correlates with accent, so
these exclusions are non-random. Highest exclusion rate: Spanish at 3.1% (22 of its 27
exclusions are code-switching); every other group below 1.4%. No group loses more than
the 10% that §4.3 sets as the flag threshold.

**AfriSpeech.** 173 of 8,400 training utterances (2.1%) were dropped for exceeding
Whisper's 30-second encoder window — hausa 65, yoruba 56, igbo 44, swahili 8, **zulu 0**.
The smallest group is untouched, so the 11.5× group imbalance under test is preserved.

**Long audio is handled differently across backends.** NeMo models chunk utterances
above 120 s (16 of 9,177; conformer attention is O(T²) and a 536 s utterance faults an
8 GB GPU), while Whisper models use native sequential long-form decoding with full
context. NeMo models are mildly disadvantaged on 0.17% of the data.

---

## 4. Efficiency measurements

**One service model, one machine.** Latency assumes a single model instance serving
requests serially — every backend serializes on a per-instance lock. A deployment with
several replicas, or aggressive server-side batching, would see different tail
behaviour. All figures come from one RTX 4060 Laptop: the *ordering* of models should
be stable, the absolute numbers are not portable to datacentre hardware.

**Moonshine's throughput is not comparable to the rest.** Its reportable runtime is
CPU-only ONNX. The CUDA/CPU boundary — not a configuration choice — separates it from
every other row, so its position on the frontier plot is indicative only.

**Thermal steady state is unverified for the fastest models.** Parakeet's timing phases
last 10–18 s, too few 1 Hz samples for a throttling verdict, and the CPU-only Moonshine
run has no CPU thermal record at all. Temperature, power and clock-band evidence is
consistent with steady state, but the protocol's own check returns *indeterminate*
there rather than *clean*.

---

## 5. Model and training caveats

**Training-data contamination is unknown.** Neither NVIDIA nor OpenAI fully discloses
training corpora, so overlap with EdAcc or AfriSpeech cannot be ruled out.
Contamination would *shrink* measured disparities, making these figures a lower bound.

**Moonshine Streaming fails the §4.5 determinism gate.** Identical audio yields
different transcriptions depending on process state. The band is measured (10.0%
utterance disagreement, |ΔWER| ±0.21 points overall, ±1.02 worst audited group) and
travels with every number reported for it.

**Moonshine's runtime differs in two further ways**: VAD re-segmentation, and a
`max_tokens_per_second` anti-loop heuristic. Both are documented defaults, but its
hallucination-loop count and, to a lesser degree, its WER reflect library policy as
well as model behaviour.

**One Part 3 training run diverged.** The τ=1.0 arm hit a transient fp16 divergence at
step ~3700 and recovered by 4400; its step4200 snapshot is unusable and self-excludes
from selection. Not re-run, because the seed is fixed across arms and a re-run would
either reproduce it or break seed-parity. See EVAL_SPEC changelog 2026-08-13.

**Validation worst-group WER did not transfer to test in Part 3.** ERM's `final`
checkpoint won on validation worst-group (24.44) but placed second-worst on test
(25.93) — expected when the worst group's validation set is 159 utterances from 22
speakers, and a caution against over-reading any single checkpoint selection.

**Worst/best ratio is post-hoc.** Computed after seeing results, exploratory, and not
used for any statistical claim.

---

## 6. Scope of the fine-tuning experiment

**One base model, one corpus, one architecture.** Part 3 fine-tunes `whisper-small`
only. Whether Group-DRO behaves differently on a transducer, a larger model, or a
corpus with different group structure is untested.

**Budget is 22 h of audio, subsampled from 102 h.** Proportional and seed-fixed, chosen
so a 4-run sweep fits on an 8 GB laptop. A larger budget could change the outcome;
identical budgets across arms is what makes the ERM/DRO comparison valid, not the
absolute size.

**Group-DRO required three implementation corrections** before it behaved as intended
(frequency capture, cumulative collapse, scale annihilation — EVAL_SPEC changelog
2026-08-12). The final formulation is stationary and scale-invariant, which departs
from the textbook online exponentiated-gradient rule §6 originally specified. A reader
who considers the textbook form the object of study should treat the null accordingly.
