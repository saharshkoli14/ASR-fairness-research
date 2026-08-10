"""Paired model comparison (EVAL_SPEC §4.4).

Overlapping per-model CIs are NOT a valid test of a difference. To claim model A
has a larger accent gap than model B, resample speakers ONCE per bootstrap
iteration and recompute BOTH models on that same resample — the paired difference
cancels the shared speaker-sampling variance.
"""

from collections import defaultdict

import numpy as np

from .metrics import BOOTSTRAP_N, BOOTSTRAP_SEED, Utterance, _counts, _metric_from_counts


def _speaker_counts(utts: list[Utterance]) -> dict[str, dict[str, tuple[int, int]]]:
    """group -> speaker -> (errors, ref_words)"""
    out: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for u in utts:
        e, w = _counts(u.ref, u.hyp)
        rec = out[u.group][u.speaker]
        rec[0] += e
        rec[1] += w
    return {g: {s: (v[0], v[1]) for s, v in d.items()} for g, d in out.items()}


def paired_bootstrap(utts_a: list[Utterance], utts_b: list[Utterance],
                     metric: str = "gap_max_minus_min",
                     n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
                     alpha: float = 0.05) -> dict:
    """CI on (metric_A - metric_B) using a shared speaker resample each iteration."""
    a, b = _speaker_counts(utts_a), _speaker_counts(utts_b)
    groups = sorted(set(a) & set(b))
    if not groups:
        raise ValueError("no shared groups between the two models")

    # Speaker lists must match per group so the resample indexes the same speakers.
    speakers = {g: sorted(set(a[g]) & set(b[g])) for g in groups}

    def point(counts):
        return _metric_from_counts(counts, metric)

    def collect(src, g, idx):
        e = sum(src[g][speakers[g][i]][0] for i in idx)
        w = sum(src[g][speakers[g][i]][1] for i in idx)
        return e, w

    full = {g: list(range(len(speakers[g]))) for g in groups}
    point_a = point({g: collect(a, g, full[g]) for g in groups})
    point_b = point({g: collect(b, g, full[g]) for g in groups})

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idxs = {g: rng.integers(0, len(speakers[g]), len(speakers[g])) for g in groups}
        ca = {g: collect(a, g, idxs[g]) for g in groups}
        cb = {g: collect(b, g, idxs[g]) for g in groups}
        diffs.append(point(ca) - point(cb))

    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "metric": metric,
        "point_a": float(point_a),
        "point_b": float(point_b),
        "difference": float(point_a - point_b),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool(lo > 0 or hi < 0),  # CI excludes zero
        "n_boot": n_boot,
        "seed": seed,
        "n_speakers_per_group": {g: len(speakers[g]) for g in groups},
    }
