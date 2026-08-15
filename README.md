## 🎙️ Accent Fairness in the 2026 ASR Generation

**TABLE OF CONTENTS**

* [Project Overview](#project-overview)
* [Objectives](#objectives)
* [Repository Structure](#️-repository-structure)
* [System Architecture](#system-architecture)
* [Installation](#️-installation)
* [Create a Virtual Environment and activate it](#create-a-virtual-environment-and-activate-it)
* [Tech Stack](#tech-stack)
* [Deliverables](#deliverables)
* [Quick Start](#quick-start)
* [Results at a Glance](#results-at-a-glance)
* [Key Insights & Outcomes](#key-insights--outcomes)
* [Limitations](#️-limitations)
* [License](#license)

---

## 📌 Project Overview

This project audits **seven current speech-recognition models for accent disparity** on a
methodology frozen *before* any model produced a number. It asks three questions the field
usually treats separately: how much does recognition accuracy differ across English accents,
does making models faster make them less fair, and can the disparity be trained away.

Every model is scored on the same cleaned benchmark with the same vendored text normalizer,
speaker-level bootstrap confidence intervals, and a determinism gate. Every departure from the
frozen methodology carries a dated changelog entry written before the affected numbers existed.

Full numbers in [`RESULTS.md`](RESULTS.md) · methodology in [`EVAL_SPEC.md`](EVAL_SPEC.md) ·
what the results cannot support in [`LIMITATIONS.md`](LIMITATIONS.md).

---

## 🎯 Objectives

* **Measure** per-group WER across 7 accent groups of EdAcc (5,087 scored utterances) with
  speaker-level bootstrap CIs and Bonferroni correction across 15 model comparisons.
* **Gate** every model on reproducibility before reporting it — identical audio must yield
  identical text, and models that fail get a measured non-determinism band attached.
* **Benchmark** throughput, tail latency under load, and VRAM on identical hardware, to test
  whether the field's move toward faster models carries a hidden fairness cost.
* **Mitigate** — full fine-tuning of `whisper-small` on AfriSpeech-200, ERM versus Group-DRO,
  evaluated both in-corpus and cross-corpus back onto EdAcc.
* **Recommend** a disparity metric on statistical grounds rather than convention.

---

## 🏗️ Repository Structure

```
asr-fairness-audit/
│── README.md
│── EVAL_SPEC.md                    # methodology, frozen before any run; dated changelog
│── RESULTS.md                      # all numbers: accuracy, efficiency, mitigation, appendix
│── LIMITATIONS.md                  # what the results cannot support
│── LICENSE
│── .gitignore
│── pyproject.toml                  # dependencies + optional extras
│── pins.json                       # model + dataset SHAs, library versions at pin time
│── groups.json                     # frozen EdAcc accent groups (7 audited, 17 pooled)
│── groups_afrispeech.json          # frozen AfriSpeech groups (5 audited of ~120 accents)
│── pin.py                          # resolves and writes pins.json
│── feasibility.py                  # pre-audit VRAM/throughput probe
│
│── src/asr_fairness_audit/
│   ├── __init__.py                 # model registry + get_transcriber (--checkpoint override)
│   ├── metrics.py                  # per-group WER, worst-group, gap, speaker bootstrap
│   ├── normalize.py                # vendored Whisper normalizer + EdAcc marker rules
│   ├── compare.py                  # paired model comparison, Bonferroni correction
│   ├── efficiency.py               # RTFx, closed-loop latency, GPU sampling, throttle verdicts
│   ├── provenance.py               # harness commit + library/CUDA versions per run
│   ├── _vendor/                    # pinned Whisper normalizer — upgrades cannot move numbers
│   ├── data/edacc.py               # EdAcc loading, cleaning rules, exclusion reporting
│   └── backends/
│       ├── base.py                 # Transcriber protocol + SerializedInference
│       ├── hf.py                   # transformers pipeline: Whisper family, Distil-Whisper
│       ├── nemo.py                 # Parakeet TDT (ASRModel), Canary-Qwen (SALM)
│       └── moonshine.py            # official moonshine-voice ONNX runtime
│
│── scripts/
│   ├── make_groups.py              # ── Part 1: freeze EdAcc accent groups
│   ├── vendor_normalizer.py        #    pin the text normalizer into _vendor/
│   ├── run_audit.py                #    the audit runner (--checkpoint for fine-tuned models)
│   ├── verify_determinism.py       #    determinism gate
│   ├── measure_nondeterminism.py   #    band for models that fail the gate
│   ├── compare_models.py           #    paired bootstrap across models
│   ├── verify_scoring_consistency.py #  re-score every model in one interpreter
│   ├── run_efficiency.py           # ── Part 2: RTFx, tail latency, VRAM, thermal record
│   ├── plot_frontier.py            #    efficiency-vs-disparity frontier
│   ├── stamp_efficiency_provenance.py # backfill provenance into early runs
│   ├── fetch_afrispeech.py         # ── Part 3: size-then-fetch, per-accent, resumable
│   ├── make_groups_afrispeech.py   #    freeze AfriSpeech groups
│   ├── prepare_afrispeech.py       #    val carve, subsample, extract, resample to 16 kHz
│   ├── verify_prepared.py          #    speaker-disjointness and manifest invariants
│   ├── train_afrispeech.py         #    ERM and Group-DRO arms, one shared loop
│   ├── eval_finetuned.py           #    scoring, checkpoint and sweep selection
│   ├── compare_arms.py             #    paired speaker bootstrap between arms
│   └── (investigations)            #    10 one-off scripts kept as evidence for changelog claims
│
│── tests/                          # metrics, normalizer golden set, efficiency regressions
│── results/
│   ├── <model>/                    # transcripts.jsonl, summary.json, efficiency.json
│   ├── frontier.png                # efficiency-vs-disparity plot
│   ├── ft/                         # fine-tuning runs, evals, selection, arm comparison
│   └── ft-{erm,dro}-edacc/         # cross-corpus audits of the fine-tuned models
```

Model weights, audio and checkpoints are gitignored; **every number in `RESULTS.md` is
re-derivable from the committed `transcripts.jsonl` files.**

---

## 🧭 System Architecture

```
   EdAcc test split                        AfriSpeech-200
   (pinned revision)                       (pinned revision)
          │                                       │
          ▼                                       ▼
   ┌─────────────┐                        ┌────────────────┐
   │  cleaning   │ marker rules,          │   prepare      │ group freeze, val carve,
   │  + groups   │ inclusion rule         │   subset       │ subsample, 16 kHz cache
   └──────┬──────┘                        └───────┬────────┘
          │                                       │
          ▼                                       ▼
   ┌─────────────────────────┐            ┌────────────────┐
   │  backends (hf / nemo /  │            │  train: ERM |  │
   │  moonshine) — one call  │            │  Group-DRO     │
   │  per instance           │            │  (shared loop) │
   └──────┬──────────────────┘            └───────┬────────┘
          │                                       │
          ▼                                       ▼
   ┌──────────────────────────────────────────────────────┐
   │  vendored normalizer → metrics → speaker bootstrap   │
   └──────┬───────────────────────────────┬───────────────┘
          │                               │
          ▼                               ▼
   accuracy + efficiency            in-corpus + cross-corpus
   (RESULTS §1–4)                   mitigation (RESULTS §5)
```

Both corpora flow through the **same** normalizer, metrics and bootstrap code, so AfriSpeech
numbers sit on the same footing as EdAcc's rather than coming from a parallel implementation.

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/asr-fairness-audit.git
cd asr-fairness-audit
```

### Create a Virtual Environment and activate it

```bash
python3 -m venv venv
source venv/bin/activate      # On macOS/Linux
venv\Scripts\activate         # On Windows
```

2. Install the package and the extras you need:

```bash
pip install -e ".[dev]"        # core + pytest/ruff
pip install -e ".[nemo]"       # Parakeet, Canary-Qwen — Linux/WSL2 only
pip install -e ".[moonshine]"  # official CPU-only ONNX runtime
pip install -e ".[finetune]"   # Part 3 — bitsandbytes 8-bit Adam, required to fit 8 GB
pip install -e ".[plots]"      # matplotlib, for the frontier plot
```

---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| **Programming** | Python 3.10+ |
| **Models** | Whisper family, Distil-Whisper, Parakeet TDT v2/v3, Canary-Qwen 2.5B, Moonshine Streaming |
| **Runtimes** | PyTorch + transformers, NVIDIA NeMo, moonshine-voice (ONNX Runtime) |
| **Audio** | soundfile, soxr (explicit resampling — avoids a hidden ffmpeg dependency) |
| **Metrics** | jiwer, vendored Whisper `EnglishTextNormalizer`, NumPy speaker-level bootstrap |
| **Training** | bitsandbytes 8-bit Adam, fp16 autocast, gradient checkpointing |
| **Plots / Tests** | matplotlib, pytest, ruff |
| **Hardware** | RTX 4060 Laptop (8 GB), 83 W cap, AC power, idle machine |

---

## 📦 Deliverables

✅ Frozen evaluation specification with a dated deviation changelog (`EVAL_SPEC.md`)
✅ Seven-model accent audit with speaker-level statistics (`RESULTS.md` §1–3)
✅ Efficiency benchmark and efficiency-vs-disparity frontier (`RESULTS.md` §4, `results/frontier.png`)
✅ Fine-tuning study: ERM vs Group-DRO, in-corpus and cross-corpus (`RESULTS.md` §5)
✅ Methodological recommendation on disparity metrics (`RESULTS.md` §6)
✅ Committed transcripts for every run — all numbers independently re-derivable
✅ Limitations document (`LIMITATIONS.md`)

---

## 🚀 Quick Start

Reproduce the headline table from the committed transcripts — no GPU, no model downloads:

```bash
python scripts/verify_scoring_consistency.py
```

This will:

* ✅ Re-score all seven models from `results/<model>/transcripts.jsonl`
* ✅ Apply the identical vendored normalizer and metrics used in the audit
* ✅ Confirm every per-group WER, worst-group, gap and speaker count reproduces exactly
* ✅ Complete in about a minute on CPU

Or run the full pipeline:

```bash
python pin.py                          # resolve checkpoint + dataset revisions
python scripts/vendor_normalizer.py    # freeze the text normalizer
python scripts/make_groups.py          # freeze accent groups from the pinned split

# Part 1 — accuracy
python scripts/run_audit.py --model parakeet-tdt-0.6b-v3
python scripts/verify_determinism.py parakeet-tdt-0.6b-v3
python scripts/compare_models.py --bonferroni --metric worst_group_wer

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
python scripts/run_audit.py --model whisper-small \
    --checkpoint results/ft/erm/step4200 --tag ft-erm-edacc      # cross-corpus
```

---

## 📊 Results at a Glance

Per-group WER (%) on the EdAcc test split, lower is better:

| Model | Params | mean | worst | gap | RTFx | p95 @ c=4 |
|---|---|---|---|---|---|---|
| **Parakeet TDT v3** | 0.6B | **15.2** | **22.2** | 13.6 | 39.5 | 0.82 s |
| Parakeet TDT v2 | 0.6B | 16.6 | 23.6 | 14.0 | **41.7** | 0.72 s |
| Distil-Whisper v3.5 | 756M | 18.5 | 25.8 | 15.3 | 17.9 | 1.98 s |
| Whisper large-v3-turbo | 809M | 19.9 | 28.4 | 17.9 | 15.1 | 2.32 s |
| Canary-Qwen | 2.5B | 21.3 | 28.7 | **13.5** | 7.3 | 5.26 s |
| Moonshine Streaming ᴺᴰ | 245M | 26.8 | 39.6 | 23.1 | 2.1 ᶜᵖᵘ | 15.6 s |
| Whisper-small † | 244M | 80.3 | 104.6 | 70.4 | 12.4 | 3.67 s |

ᴺᴰ fails the determinism gate; band measured and attached · ᶜᵖᵘ CPU-only ONNX, throughput not
comparable to the CUDA rows · † the fine-tuning base for Part 3, not a deployment candidate

---

## 📈 Key Insights & Outcomes

**Efficiency does not cost accent fairness — edge optimisation does.** The Pareto frontier over
(throughput, worst-group WER) contains only the two Parakeet models; every other CUDA model is
*strictly dominated on both axes*. Parakeet v3 is 2.6× faster than the incumbent at a third of
its p95 latency under load, and fairer. The one model fitting the "faster is less fair"
hypothesis, Moonshine Streaming, is both the slowest measured and the least fair — pointing at
streaming/edge architecture rather than efficiency as the cost driver.

**Leaderboard rank does not transfer to accented speech.** Canary-Qwen 2.5B leads the Open ASR
Leaderboard (~5.6% WER) but places 5th of 7 here; Parakeet TDT 0.6B beats it by 6 points.

**The max−min gap is the weakest disparity metric in common use.** On identical data under
identical correction it reaches significance in 4/15 pairwise comparisons against worst-group
WER's 10/15. Report worst-group WER as primary.

**Fine-tuning cuts WER 43% and leaves the disparity untouched.** Macro WER 34.9 → 19.7 while the
gap barely moves (9.89 → 9.34). Per-group improvement **rank-orders perfectly with each group's
training hours** — Yoruba (10.1 h) gains 17.9 points, Zulu (0.9 h) gains 11.9.

**Group-DRO did not fix it.** Swept over three weight-tilt strengths spanning 1.5×–5× with
identical budgets, it is statistically indistinguishable from ERM: Δ worst-group −0.43
[−1.69, +3.48], paired speaker bootstrap over 403 speakers, no dose-response.

**The mitigation does not transfer — it inverts.** Both fine-tuned models are worse on *every*
EdAcc group than the base they started from, and the gap widens by two-thirds (70.4 → 117.3).
The arm selected for fairness transferred worst.

### How this repo treats correctness

Several findings came from checking what the code actually did rather than what it appeared to:

* **Efficiency latency runs were discarded** after the NeMo backends were found to be losing
  every concurrent worker to a thread-safety fault, silently reporting 117 of 120 requests as a
  complete sweep.
* **Group-DRO needed three corrections** before it optimised what it claimed to — chasing group
  *frequency* under natural sampling, then cumulative *history*, then annihilating its own tilt
  as losses converged. Each version would have produced plausible-looking numbers.
* **All seven models were re-scored in a single interpreter** after per-run provenance revealed
  the audit spanned two environments with different major versions of `jiwer`. They reproduce
  exactly (max |Δ| = 0).

---

## ⚠️ Limitations

* EdAcc speaker counts are small (3–9 per group) — per-group WER partly reflects individuals.
* Southern British English is absent; 17 accent groups fall below the inclusion threshold.
* Cleaning is not random with respect to accent (Spanish 3.07%, all others < 1.4%).
* Moonshine's throughput is not comparable — CPU-only ONNX, a hardware boundary.
* Part 3's null is bounded: effects larger than ~2 WER points are excluded, smaller ones are not.
* Training-data contamination cannot be ruled out; it would *shrink* measured disparities.

Full treatment in [`LIMITATIONS.md`](LIMITATIONS.md).

---

## 📜 License

Released under the **MIT License** — see [`LICENSE`](LICENSE).

EdAcc and AfriSpeech-200 carry their own licences; AfriSpeech-200 is **CC-BY-NC-SA-4.0**
(non-commercial), which governs any redistribution of derived audio or fine-tuned weights.
