"""Disparity metrics (EVAL_SPEC §4.1, §4.4).

Inputs are ALREADY-NORMALIZED (ref, hyp) pairs with group and speaker labels.
Bootstrap resamples SPEAKERS (utterances within a speaker are correlated).
"""

from collections import defaultdict
from dataclasses import dataclass

import jiwer
import numpy as np

BOOTSTRAP_SEED = 3407
BOOTSTRAP_N = 1000


@dataclass
class Utterance:
    ref: str
    hyp: str
    group: str
    speaker: str


def _counts(ref: str, hyp: str) -> tuple[int, int]:
    """(errors, ref_words) for one pair."""
    if not ref.strip():
        return (len(hyp.split()), 0)  # empty ref: all hyp words are insertions
    out = jiwer.process_words(ref, hyp)
    errors = out.substitutions + out.deletions + out.insertions
    words = out.hits + out.substitutions + out.deletions
    return errors, words


def evaluate(utts: list[Utterance]) -> dict:
    """Per-group WER + micro/macro mean, worst-group, gap, std."""
    per_utt = [(u, *_counts(u.ref, u.hyp)) for u in utts]

    group_err, group_words, group_speakers, group_n = (
        defaultdict(int), defaultdict(int), defaultdict(set), defaultdict(int))
    tot_err = tot_words = 0
    for u, err, words in per_utt:
        group_err[u.group] += err
        group_words[u.group] += words
        group_speakers[u.group].add(u.speaker)
        group_n[u.group] += 1
        tot_err += err
        tot_words += words

    per_group = {
        g: {
            "wer": group_err[g] / group_words[g] if group_words[g] else float("nan"),
            "n_utterances": group_n[g],
            "n_speakers": len(group_speakers[g]),
            "ref_words": group_words[g],
        }
        for g in group_err
    }
    wers = [v["wer"] for v in per_group.values()]
    return {
        "micro_wer": tot_err / tot_words if tot_words else float("nan"),
        "macro_wer": float(np.mean(wers)),
        "worst_group_wer": float(np.max(wers)),
        "worst_group": max(per_group, key=lambda g: per_group[g]["wer"]),
        "gap_max_minus_min": float(np.max(wers) - np.min(wers)),
        "std_across_groups": float(np.std(wers, ddof=0)),
        "per_group": per_group,
    }


def _metric_from_counts(counter: dict, metric: str) -> float:
    wers = [e / w for e, w in counter.values() if w > 0]
    if not wers:
        return float("nan")
    if metric == "worst_group_wer":
        return max(wers)
    if metric == "gap_max_minus_min":
        return max(wers) - min(wers)
    if metric == "macro_wer":
        return float(np.mean(wers))
    raise ValueError(metric)


def bootstrap_ci(utts: list[Utterance], metric: str = "gap_max_minus_min",
                 n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
                 alpha: float = 0.05) -> dict:
    """Percentile CI by resampling speakers with replacement within each group."""
    rng = np.random.default_rng(seed)
    # speaker -> (group, errors, words), precomputed once
    spk: dict[str, list] = defaultdict(lambda: [None, 0, 0])
    for u in utts:
        err, words = _counts(u.ref, u.hyp)
        rec = spk[u.speaker]
        rec[0] = u.group
        rec[1] += err
        rec[2] += words
    by_group: dict[str, list] = defaultdict(list)
    for _, (g, e, w) in spk.items():
        by_group[g].append((e, w))

    stats = []
    for _ in range(n_boot):
        counter = {}
        for g, speakers in by_group.items():
            idx = rng.integers(0, len(speakers), len(speakers))
            e = sum(speakers[i][0] for i in idx)
            w = sum(speakers[i][1] for i in idx)
            counter[g] = (e, w)
        stats.append(_metric_from_counts(counter, metric))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"metric": metric, "point": _metric_from_counts(
        {g: (sum(e for e, _ in v), sum(w for _, w in v)) for g, v in by_group.items()}, metric),
        "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot, "seed": seed}
