"""Efficiency-vs-disparity frontier: worst-group WER against throughput.

    python scripts/plot_frontier.py

Reads results/<model>/{summary.json,efficiency.json} and writes results/frontier.png.

Axis choices, both deliberate:
  * x = RTFx at batch 1, the latency-relevant figure a serving deployment sees, not
    the best batched number (which rewards throughput-oriented offline use).
  * y = worst-group WER, the disparity metric §4 of RESULTS.md argues should be
    primary. Lower-right is better on both axes.

Moonshine is drawn hollow: its runtime is CPU-only ONNX, so its x position is not
comparable to the CUDA models. Whisper-small is excluded — its >100% group WERs
(hallucination loops) would compress the y axis to illegibility.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parents[1]
EXCLUDE = {"whisper-small"}          # fine-tuning base, not a deployment candidate
LABELS = {
    "parakeet-tdt-0.6b-v2": "Parakeet TDT v2",
    "parakeet-tdt-0.6b-v3": "Parakeet TDT v3",
    "distil-large-v3.5": "Distil-Whisper v3.5",
    "whisper-large-v3-turbo": "Whisper large-v3-turbo",
    "canary-qwen-2.5b": "Canary-Qwen 2.5B",
    "moonshine-streaming-medium": "Moonshine Streaming",
}


def load() -> list[dict]:
    out = []
    for d in sorted((ROOT / "results").iterdir()):
        if not d.is_dir() or d.name in EXCLUDE:
            continue
        eff_f, sum_f = d / "efficiency.json", d / "summary.json"
        if not (eff_f.exists() and sum_f.exists()):
            continue
        eff, summ = json.loads(eff_f.read_text()), json.loads(sum_f.read_text())
        out.append({
            "name": LABELS.get(d.name, d.name),
            "rtfx": eff["rtfx_batch1"],
            "wer": 100 * summ["metrics"]["worst_group_wer"],
            "cuda": eff.get("device") == "cuda",
        })
    return out


def pareto(points: list[dict]) -> list[dict]:
    """Non-dominated on (high RTFx, low worst-group WER). CUDA models only."""
    cuda = [p for p in points if p["cuda"]]
    return [p for p in cuda
            if not any(q["rtfx"] >= p["rtfx"] and q["wer"] <= p["wer"] and q is not p
                       for q in cuda)]


def main():
    pts = load()
    if not pts:
        sys.exit("no results found — run run_audit.py and run_efficiency.py first")
    front = sorted(pareto(pts), key=lambda p: p["rtfx"])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if len(front) > 1:
        ax.plot([p["rtfx"] for p in front], [p["wer"] for p in front],
                "--", color="0.6", lw=1, zorder=1, label="Pareto frontier")

    xs = [p["rtfx"] for p in pts]
    span_x, span_y = max(xs) - min(xs), max(p["wer"] for p in pts) - min(p["wer"] for p in pts)
    ax.set_xlim(min(xs) - 0.08 * span_x, max(xs) + 0.22 * span_x)

    for i, p in enumerate(pts):
        on = p in front
        ax.scatter(p["rtfx"], p["wer"], s=110, zorder=3,
                   facecolor=("#c0392b" if on else "#2c3e50") if p["cuda"] else "none",
                   edgecolor="#2c3e50", linewidth=1.5)
        # Labels sit right of the marker, except near the right edge where they would
        # run off; points crowded together get their labels dodged vertically instead.
        right_edge = p["rtfx"] > min(xs) + 0.72 * span_x
        crowded = any(q is not p and abs(q["wer"] - p["wer"]) < 0.06 * span_y
                      and abs(q["rtfx"] - p["rtfx"]) < 0.35 * span_x for q in pts)
        dx, ha = (-11, "right") if right_edge else (11, "left")
        dy = (11 if i % 2 == 0 else -15) if crowded else -3
        ax.annotate(p["name"] + ("" if p["cuda"] else "  (CPU)"),
                    (p["rtfx"], p["wer"]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=9)

    ax.set_xlabel("RTFx at batch 1  (audio seconds per wall-clock second) →  faster")
    ax.set_ylabel("Worst-group WER (%)  →  less fair")
    ax.set_title("Efficiency vs accent disparity — EdAcc test, RTX 4060 Laptop 8 GB")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.text(0.5, 0.005,
             "Lower-right is better on both axes. Moonshine (hollow) runs CPU-only ONNX — "
             "its throughput is not comparable to the CUDA models.",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = ROOT / "results" / "frontier.png"
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")
    print("Pareto-optimal (CUDA): " + ", ".join(f"{p['name']} ({p['rtfx']}x, {p['wer']:.1f}%)"
                                                for p in front))


if __name__ == "__main__":
    main()
