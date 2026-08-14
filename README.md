# Accent fairness in the 2026 ASR generation

An audit of seven current speech-recognition models for **accent disparity**, on a
methodology frozen before any model produced a number. Three parts:

1. **Accuracy** — per-group WER across 7 accent groups of EdAcc, with speaker-level
   bootstrap CIs and a determinism gate.
2. **Efficiency** — throughput, tail latency under load, and VRAM on identical hardware,
   to test whether faster models are less fair.
3. **Mitigation** — full fine-tuning of `whisper-small` on AfriSpeech-200, ERM versus
   Group-DRO, evaluated in-corpus and cross-corpus.

Full numbers in [`RESULTS.md`](RESULTS.md). Methodology and every dated deviation in
[`EVAL_SPEC.md`](EVAL_SPEC.md). What the results cannot support: [`LIMITATIONS.md`](LIMITATIONS.md).

---

## Headline findings

**Efficiency does not cost accent fairness — edge optimisation does.** The Pareto
frontier over (throughput, worst-group WER) contains only the two Parakeet TDT models:
every other CUDA model is *strictly dominated* on both axes. Parakeet v3 is 2.6× faster
than the incumbent Whisper large-v3-turbo at a third of its p95 latency under load, and
lower on worst-group WER. The one model fitting the "faster is less fair" hypothesis —
Moonshine Streaming — is both the slowest measured and the least fair, pointing at
streaming/edge architecture rather than efficiency as the cost driver.

**Leaderboard rank does not transfer to accented speech.** Canary-Qwen 2.5B leads the
Open ASR Leaderboard (~5.6% WER) but places 5th of 7 here; Parakeet TDT 0.6B beats it by
6 points on EdAcc. Leaderboard position is not evidence of accent robustness.

**The max−min gap is the weakest disparity metric in common use.** On identical data
under identical correction it reaches significance in 4/15 pairwise comparisons against
worst-group WER's 10/15. Report worst-group WER as primary; the gap is a difference of
two noisy estimates and carries roughly double the variance.

**Fine-tuning cuts WER by 43% and leaves the disparity untouched.** ERM fine-tuning on
AfriSpeech moves macro WER 34.9 → 19.7 while the gap barely moves (9.89 → 9.34). The
per-group improvement **rank-orders perfectly with each group's training hours** —
Yoruba (10.1 h) gains 17.9 points, Zulu (0.9 h) gains 11.9.

**Group-DRO did not fix it.** Swept over three weight-tilt strengths spanning 1.5×–5×,
with identical budgets and per-arm selection criteria, Group-DRO is statistically
indistinguishable from ERM: Δ worst-group −0.43 [−1.69, +3.48], Δ gap −0.69 [−2.30,
+3.21], paired speaker bootstrap over 403 speakers. No dose-response across the sweep.
The disparity is inherited from pretraining — the base model already ranked the groups
in the same order — and neither arm changes that ordering.

---

## Reproducing

```bash
pip install -e ".[dev]"
python pin.py                          # resolve checkpoint + dataset revisions
python scripts/vendor_normalizer.py    # freeze the text normalizer
python scripts/make_groups.py          # freeze accent groups from the pinned split

# Part 1 — accuracy
python scripts/run_audit.py --model parakeet-tdt-0.6b-v3
python scripts/verify_determinism.py parakeet-tdt-0.6b-v3
python scripts/compare_models.py --bonferroni --metric worst_group_wer
python scripts/verify_scoring_consistency.py     # all models re-scored in one interpreter

# Part 2 — efficiency
python scripts/run_efficiency.py --model parakeet-tdt-0.6b-v3
python scripts/plot_frontier.py

# Part 3 — fine-tuning
python scripts/fetch_afrispeech.py --get transcripts dev test --dest <data-dir>
python scripts/make_groups_afrispeech.py --data-dir <data-dir>
python scripts/prepare_afrispeech.py --data-dir <data-dir>
python scripts/verify_prepared.py --data-dir <data-dir>
python scripts/train_afrispeech.py --data-dir <data-dir> --arm erm --steps 16800
python scripts/eval_finetuned.py --data-dir <data-dir> --ckpt results/ft/erm --all --split val
python scripts/compare_arms.py --a <erm transcripts> --b <dro transcripts>
```

NeMo models (Parakeet, Canary-Qwen) require Linux or WSL2: `pip install "nemo_toolkit[asr]"`.
Moonshine uses the official ONNX runtime: `pip install moonshine-voice`.

Hardware for every run: RTX 4060 Laptop (8 GB), 83 W cap, AC power, idle machine.

---

## How this repo treats correctness

The methodology was frozen in `EVAL_SPEC.md` before any model was run, and every
departure from it carries a dated changelog entry written *before* the affected numbers
were produced. Several were found by checking what the code actually did rather than
what it appeared to do:

- Efficiency latency runs were **discarded** after the NeMo backends were found to be
  losing every concurrent worker to a thread-safety fault, silently reporting 117 of 120
  requests as a complete sweep.
- Group-DRO needed **three** corrections before it optimised what it claimed to —
  chasing group *frequency* under natural sampling, then cumulative *history*, then
  annihilating its own tilt as losses converged. Each version would have produced
  plausible-looking numbers.
- All seven models were **re-scored in a single interpreter** after per-run provenance
  revealed the audit spanned two environments with different major versions of `jiwer`.
  They reproduce exactly (max |Δ| = 0).

Artifacts: `pins.json` (model/dataset SHAs), `groups.json` and `groups_afrispeech.json`
(frozen group definitions), committed `transcripts.jsonl` for every run, and a
provenance block (commit, dirty flag, library versions, CUDA) in each results file.

## License

MIT. EdAcc and AfriSpeech-200 carry their own licences; AfriSpeech-200 is CC-BY-NC-SA-4.0.
