"""Score a fine-tuned (or base) Whisper on the prepared AfriSpeech splits (EVAL_SPEC §6).

    # one checkpoint
    python scripts/eval_finetuned.py --data-dir /mnt/c/asr-data/afrispeech-200 \
        --ckpt results/ft/erm/final --split val

    # every snapshot in a run directory, for checkpoint selection
    python scripts/eval_finetuned.py --data-dir ... --ckpt results/ft/erm --all --split val

    # the un-fine-tuned baseline — the "before" the experiment is measured against
    python scripts/eval_finetuned.py --data-dir ... --ckpt base --split test --bootstrap

    # apply §6's selection rule to everything scored so far
    python scripts/eval_finetuned.py --select results/ft/erm --arm erm

Scoring is identical to the EdAcc audit: same vendored normalizer, same
`metrics.evaluate`, same speaker-level bootstrap — so AfriSpeech numbers are
comparable to Part 1's rather than computed by a parallel code path.

Decoding is batch 1 by default, matching §5: batched padding changes Whisper
outputs (EVAL_SPEC changelog 2026-08-07). `--batch >1` is permitted only for
checkpoint *selection* on val, is recorded in the output, and must never produce
a reported number.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit import load_pins  # noqa: E402
from asr_fairness_audit.metrics import Utterance, bootstrap_ci, evaluate  # noqa: E402
from asr_fairness_audit.normalize import normalize, normalize_reference, vendor_info  # noqa: E402
from asr_fairness_audit.provenance import run_provenance  # noqa: E402

BASE = "openai/whisper-small"


def load_split(prep: Path, split: str) -> list[dict]:
    rows = [json.loads(x) for x in
            (prep / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if r["split"] == split]


def transcribe(ckpt: str, rows: list[dict], prep: Path, batch: int) -> list[str]:
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    src = BASE if ckpt == "base" else ckpt
    rev = load_pins()["models"].get(BASE) if ckpt == "base" else None
    proc = WhisperProcessor.from_pretrained(BASE, revision=load_pins()["models"].get(BASE),
                                            language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(src, revision=rev)
    model = model.to("cuda").eval().half()
    forced = proc.get_decoder_prompt_ids(language="en", task="transcribe")

    out, t0 = [], time.time()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        audio = [sf.read(prep / r["path"].replace("\\", "/"), dtype="float32")[0] for r in chunk]
        feats = proc.feature_extractor(audio, sampling_rate=16_000,
                                       return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(feats.to("cuda").half(), forced_decoder_ids=forced,
                                 max_new_tokens=440)
        out += proc.batch_decode(ids, skip_special_tokens=True)
        if (i // max(batch, 1)) % 25 == 0:
            done = min(i + batch, len(rows))
            rate = done / max(time.time() - t0, 1e-9)
            print(f"\r  {done}/{len(rows)}  {rate:.1f} utt/s  "
                  f"~{(len(rows) - done) / max(rate, 1e-9) / 60:.0f} min left", end="", flush=True)
    print()
    return out


def score(ckpt: str, split: str, prep: Path, batch: int, bootstrap: bool, out_dir: Path) -> dict:
    rows = load_split(prep, split)
    print(f"\n{ckpt}  split={split}  {len(rows)} utterances  batch={batch}")
    hyps = transcribe(ckpt, rows, prep, batch)

    utts, loops = [], {}
    for r, hyp in zip(rows, hyps):
        ref_n, hyp_n = normalize_reference(r["text"]), normalize(hyp)
        if len(hyp_n.split()) > max(10, 5 * len(ref_n.split())):
            loops.setdefault(r["accent"], []).append(r["path"])
        utts.append(Utterance(ref=ref_n, hyp=hyp_n, group=r["accent"], speaker=r["speaker"]))

    res = {
        "checkpoint": ckpt, "split": split, "n": len(utts), "batch_size": batch,
        "batch_caveat": None if batch == 1 else
        "batched padding can change Whisper outputs (EVAL_SPEC 2026-08-07) — selection only",
        "normalizer_source": vendor_info()["commit"],
        "metrics": evaluate(utts),
        "hallucination_loops_by_group": {g: len(v) for g, v in loops.items()},
        "provenance": run_provenance(),
    }
    if bootstrap:
        res["bootstrap"] = {m: bootstrap_ci(utts, metric=m)
                            for m in ("gap_max_minus_min", "worst_group_wer", "macro_wer")}

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "base" if ckpt == "base" else Path(ckpt).name
    (out_dir / f"eval_{split}_{tag}.json").write_text(json.dumps(res, indent=2))
    with (out_dir / f"transcripts_{split}_{tag}.jsonl").open("w", encoding="utf-8") as fh:
        for r, hyp in zip(rows, hyps):
            fh.write(json.dumps({"path": r["path"], "accent": r["accent"],
                                 "speaker": r["speaker"], "ref": r["text"], "hyp": hyp}) + "\n")

    m = res["metrics"]
    print(f"  macro {100 * m['macro_wer']:.1f}  worst {100 * m['worst_group_wer']:.1f} "
          f"({m['worst_group']})  gap {100 * m['gap_max_minus_min']:.1f}")
    for g, s in sorted(m["per_group"].items(), key=lambda kv: -kv[1]["wer"]):
        print(f"    {g:10} {100 * s['wer']:6.1f}  ({s['n_utterances']} utt, {s['n_speakers']} spk)")
    return res


def select_sweep(run_dirs: list[Path], arm: str, out: Path) -> None:
    """§6: pick the (hyperparameter, checkpoint) pair, not just the checkpoint.

    The DRO step size is swept on validation worst-group WER, so the winner has to
    be chosen across tau values as well as across snapshots — selecting the best
    checkpoint within each run and then eyeballing the three would leave the
    hyperparameter choice undocumented.
    """
    key = "macro_wer" if arm == "erm" else "worst_group_wer"
    rows = []
    for d in run_dirs:
        cfg = json.loads((d / "run_config.json").read_text()) if (d / "run_config.json").exists() else {}
        for f in sorted(d.glob("eval_val_*.json")):
            e = json.loads(f.read_text())
            rows.append({"run": d.name, "tau": cfg.get("dro_eta"), "ckpt": e["checkpoint"],
                         "steps": int(Path(e["checkpoint"]).name[4:])
                         if Path(e["checkpoint"]).name.startswith("step") else 10**9,
                         "m": e["metrics"]})
    if not rows:
        sys.exit("no eval_val_*.json found in " + ", ".join(str(d) for d in run_dirs))

    TIE_BAND = 0.0005
    rows.sort(key=lambda r: (r["m"][key], r["steps"]))
    best_val = rows[0]["m"][key]
    tied = [r for r in rows if abs(r["m"][key] - best_val) <= TIE_BAND]
    best = min(tied, key=lambda r: r["steps"]) if len(tied) > 1 else rows[0]

    print(f"sweep selection for arm={arm} by validation {key}\n")
    print(f"{'run':16} {'tau':>6} {'ckpt':>10} {'worst':>7} {'macro':>7} {'gap':>7}")
    for r in rows:
        mark = "  <- selected" if r is best else ""
        print(f"{r['run']:16} {str(r['tau']):>6} {Path(r['ckpt']).name:>10} "
              f"{100 * r['m']['worst_group_wer']:7.2f} {100 * r['m']['macro_wer']:7.2f} "
              f"{100 * r['m']['gap_max_minus_min']:7.2f}{mark}")
    out.write_text(json.dumps(
        {"arm": arm, "criterion": f"validation {key} across sweep",
         "selected": {"run": best["run"], "tau": best["tau"], "checkpoint": best["ckpt"]},
         "tiebreak": "fewest steps within 0.05 WER points" if len(tied) > 1 else None,
         "all": [{"run": r["run"], "tau": r["tau"], "checkpoint": r["ckpt"],
                  "worst_group_wer": r["m"]["worst_group_wer"],
                  "macro_wer": r["m"]["macro_wer"],
                  "gap_max_minus_min": r["m"]["gap_max_minus_min"]} for r in rows]}, indent=2))
    print(f"\nwrote {out}")


def select(run_dir: Path, arm: str) -> None:
    """§6: ERM selected on validation MEAN WER, DRO on validation WORST-GROUP WER.

    Selecting both by mean would sandbag DRO, which trades mean for worst-group by
    construction; selecting both by worst-group would flatter it.
    """
    key = "macro_wer" if arm == "erm" else "worst_group_wer"
    evals = sorted(run_dir.glob("eval_val_*.json"))
    if not evals:
        sys.exit(f"no eval_val_*.json in {run_dir} — score some checkpoints first")
    scored = [json.loads(f.read_text()) for f in evals]

    def steps_of(d):
        name = Path(d["checkpoint"]).name
        return int(name[4:]) if name.startswith("step") else 10**9   # "final" sorts last

    # TIEBREAK (explicit, 2026-08-12): validation WER differences below 0.05 points are
    # noise at n~1300 — the 2026-08-12 ERM sweep separated its top two checkpoints by
    # 0.0009 points. Within that band, prefer FEWER training steps: cheaper, and less
    # memorised (train loss was still falling long after val WER flattened). Without a
    # stated rule the winner is decided by filesystem ordering.
    TIE_BAND = 0.0005   # 0.05 WER points, in the [0,1] units metrics use
    scored.sort(key=lambda d: (d["metrics"][key], steps_of(d)))
    best_val = scored[0]["metrics"][key]
    tied = [d for d in scored if abs(d["metrics"][key] - best_val) <= TIE_BAND]
    best = min(tied, key=steps_of) if len(tied) > 1 else scored[0]

    print(f"selection for arm={arm} by validation {key}:")
    for d in scored:
        mark = "   <- selected" if d is best else ("   (tied)" if d in tied else "")
        print(f"  {100 * d['metrics'][key]:7.4f}  {d['checkpoint']}{mark}")
    if len(tied) > 1:
        print(f"\n  {len(tied)} checkpoints within {100 * TIE_BAND:.2f} WER points — "
              f"criterion does not separate them; broke the tie on fewest steps.")
        print("  Sensitivity of the disparity numbers across the tied set:")
        for d in sorted(tied, key=steps_of):
            m = d["metrics"]
            print(f"    {Path(d['checkpoint']).name:10} worst {100 * m['worst_group_wer']:6.2f}  "
                  f"gap {100 * m['gap_max_minus_min']:6.2f}")

    (run_dir / "selected.json").write_text(json.dumps(
        {"arm": arm, "criterion": f"validation {key}", "checkpoint": best["checkpoint"],
         "tiebreak": ("fewest steps within %.2f WER points" % (100 * TIE_BAND)
                      if len(tied) > 1 else None),
         "tied_with": [d["checkpoint"] for d in tied if d is not best],
         "tied_sensitivity": {Path(d["checkpoint"]).name: {
             "worst_group_wer": d["metrics"]["worst_group_wer"],
             "gap_max_minus_min": d["metrics"]["gap_max_minus_min"]} for d in tied},
         "validation_metrics": best["metrics"]}, indent=2))
    print(f"\nwrote {run_dir / 'selected.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir")
    ap.add_argument("--ckpt", help="checkpoint dir, a run dir with --all, or 'base'")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--all", action="store_true", help="score every snapshot in the run dir")
    ap.add_argument("--bootstrap", action="store_true", help="speaker-level CIs (slow)")
    ap.add_argument("--select", help="apply §6 selection over a run dir's val evals")
    ap.add_argument("--select-sweep", nargs="*", metavar="RUN_DIR",
                    help="select the best (hyperparameter, checkpoint) pair across runs")
    ap.add_argument("--arm", choices=["erm", "dro"], default="erm")
    args = ap.parse_args()

    if args.select_sweep:
        dirs = [Path(d) for d in args.select_sweep]
        return select_sweep(dirs, args.arm, dirs[0].parent / f"selected_{args.arm}_sweep.json")
    if args.select:
        return select(Path(args.select), args.arm)
    if not (args.data_dir and args.ckpt):
        sys.exit("--data-dir and --ckpt are required (or use --select)")
    if args.batch > 1 and args.split == "test":
        sys.exit("refusing: test numbers are reported, and batching can change Whisper "
                 "outputs (EVAL_SPEC §5). Use --batch 1 for test.")

    prep = Path(args.data_dir) / "prepared"
    if args.all:
        run = Path(args.ckpt)
        ckpts = sorted([str(p) for p in run.glob("step*") if p.is_dir()],
                       key=lambda s: int(Path(s).name[4:]))
        if (run / "final").exists():
            ckpts.append(str(run / "final"))
        print(f"scoring {len(ckpts)} checkpoint(s) from {run}")
        for c in ckpts:
            score(c, args.split, prep, args.batch, args.bootstrap, run)
    else:
        out = Path(args.ckpt).parent if args.ckpt != "base" else ROOT / "results" / "ft" / "base"
        score(args.ckpt, args.split, prep, args.batch, args.bootstrap, out)


if __name__ == "__main__":
    main()
