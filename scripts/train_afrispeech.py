"""Fine-tune whisper-small on the prepared AfriSpeech subset (EVAL_SPEC §6).

    # 1. smoke test FIRST — ~10 min, proves it fits in 8 GB and the loss moves
    python scripts/train_afrispeech.py --data-dir /mnt/c/asr-data/afrispeech-200 --smoke

    # 2. overnight ERM run, resumable
    python scripts/train_afrispeech.py --data-dir /mnt/c/asr-data/afrispeech-200 \
        --arm erm --steps 6000 --out results/ft/erm

    # 3. resume after a crash / reboot — picks up from the last checkpoint
    python scripts/train_afrispeech.py --data-dir ... --arm erm --out results/ft/erm --resume

Run under WSL: bitsandbytes' 8-bit Adam has proper Linux wheels, and the NeMo env
already carries a CUDA torch.

ARMS
  erm  — mean loss over the batch (the baseline §6 compares against).
  dro  — group-weighted loss; weights are a softmax over each group's current EMA
         loss, q_g ∝ exp(Lbar_g / tau), with tau = --dro-eta. Stationary by design:
         the cumulative exponentiated-gradient form collapses onto one group under
         this corpus's 11.5x imbalance (EVAL_SPEC changelog 2026-08-12).

Both arms share this loop deliberately: §6 requires identical budgets and schedules,
and sharing the code path makes that verifiable rather than asserted. The only
difference is how per-sample losses are reduced (see `reduce_loss`).

MEMORY (8 GB): fp16 autocast + gradient checkpointing + 8-bit Adam. whisper-small is
244M params: fp32 master weights ~1 GB, grads ~1 GB, 8-bit Adam states ~0.5 GB
(vs ~2 GB for fp32 Adam — this is what makes it fit). Batch 4 with grad-accum 4
gives an effective batch of 16.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_fairness_audit import load_pins  # noqa: E402
from asr_fairness_audit.provenance import run_provenance  # noqa: E402

SEED = 3407
BASE = "openai/whisper-small"


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


MAX_LABEL_TOKENS = 448      # Whisper max_target_positions — a hard model limit
MAX_AUDIO_S = 30.0          # Whisper's encoder window


class AfriSpeech(Dataset):
    """Prepared 16 kHz FLAC + transcript, tagged with its accent group.

    Two filters, applied once at construction so a bad sample fails at startup
    rather than 4,000 steps into an unattended run:

      * duration > 30 s — the encoder only sees the first 30 s, so the reference
        describes audio the model was never shown. Training on those pairs teaches
        the decoder to invent the tail.
      * labels > 448 tokens — hard model limit; raises mid-batch otherwise.

    Transcripts are tokenized here rather than per __getitem__: it makes the length
    filter exact, and saves re-tokenizing every sample every epoch.
    """

    def __init__(self, prep: Path, split: str, processor, groups: list[str]):
        self.prep, self.processor = prep, processor
        self.gidx = {g: i for i, g in enumerate(groups)}
        rows = [json.loads(x) for x in
                (prep / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
        rows = [r for r in rows if r["split"] == split]

        self.rows, self.labels = [], []
        self.dropped = {"long_audio": 0, "long_labels": 0, "by_group": defaultdict(int)}
        for r in rows:
            if r["duration"] > MAX_AUDIO_S:
                self.dropped["long_audio"] += 1
                self.dropped["by_group"][r["accent"]] += 1
                continue
            ids = processor.tokenizer(r["text"]).input_ids
            if len(ids) > MAX_LABEL_TOKENS:
                self.dropped["long_labels"] += 1
                self.dropped["by_group"][r["accent"]] += 1
                continue
            self.rows.append(r)
            self.labels.append(ids)
        self.dropped["by_group"] = dict(self.dropped["by_group"])

    def hours(self) -> float:
        return sum(r["duration"] for r in self.rows) / 3600

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import soundfile as sf
        r = self.rows[i]
        # The manifest may have been written on Windows (backslash separators) and is
        # read here under WSL; normalise or libsndfile receives one long filename.
        audio, sr = sf.read(self.prep / r["path"].replace("\\", "/"), dtype="float32")
        feats = self.processor.feature_extractor(audio, sampling_rate=sr).input_features[0]
        return {"input_features": feats, "labels": self.labels[i],
                "group": self.gidx[r["accent"]]}


def collate(batch, pad_id: int):
    feats = torch.tensor(np.stack([b["input_features"] for b in batch]))
    n = max(len(b["labels"]) for b in batch)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, :len(b["labels"])] = torch.tensor(b["labels"])
    return {"input_features": feats, "labels": labels,
            "groups": torch.tensor([b["group"] for b in batch])}


def per_sample_loss(logits, labels):
    """Mean token CE per sample (not per token) so groups weight by utterance."""
    ce = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)).float(), labels.view(-1),
        ignore_index=-100, reduction="none").view(labels.shape)
    mask = (labels != -100).float()
    return (ce * mask).sum(1) / mask.sum(1).clamp(min=1)


Q_FLOOR = 1e-3
EMA_BETA = 0.95     # per-group loss tracker; ~20-batch memory per group


def reduce_loss(losses, groups, arm, q, eta, n_groups, ema=None):
    """ERM: plain mean. DRO: group-weighted, exponentiated-gradient update on q.

    Three details that matter, all found empirically on 2026-08-12:

    * **Weights are STATIONARY, not cumulative.** The textbook online rule
      q_g <- q_g * exp(eta * L_g) weights by the *running product* of past losses,
      so its concentration grows with the training horizon: at eta=0.01 an EMA
      spread of only 0.38 nats drove q(hausa) to 0.87 by step 1450 of 16800, with
      every other group at the floor. Since eta=0.01 is the smallest value §6
      specifies, the whole sweep would have produced single-group models. Here q is
      instead a softmax over the *current* per-group EMA loss, q_g ∝ exp(Lbar_g/tau):
      it responds to how hard each group is now, cannot run away with the horizon,
      and lets a group recover the moment its loss rises. `--dro-eta` is that tau.
    * **Drive it from an EMA over ALL known groups, not only this batch's.** The
      textbook update touches only present groups — fine under group-balanced
      sampling, catastrophic under natural sampling: Yoruba is 46% of this corpus
      and Zulu 4%, so Yoruba collected a boost nearly every step while Zulu was
      shrunk by renormalisation in its absence. An earlier probe collapsed onto
      q(yoruba)=0.87 — DRO chasing the *largest* group rather than the hardest.
      The EMA decouples update frequency from group frequency and leaves the
      sampler identical between arms, so ERM/DRO stays a single-variable contrast.
    * **Renormalise over the groups present in this batch.** With 5 groups and
      batch 4, most batches hold 3-4 of them; weighting by global q would make the
      batch loss sum to less than 1x the mean by a different amount every step,
      wobbling the effective learning rate. Renormalising keeps DRO on ERM's scale.
    * **Floor q.** The update is multiplicative, so a weight decaying to zero can
      never recover however badly the model later does on that group — absorbing,
      and silently degrades Group-DRO into weighted ERM.
    """
    if arm == "erm":
        return losses.mean(), q
    per_group = {int(g): losses[groups == g].mean() for g in groups.unique()}
    with torch.no_grad():
        for g, gl in per_group.items():                       # track only what we saw
            ema[g] = float(gl) if ema[g] < 0 else EMA_BETA * ema[g] + (1 - EMA_BETA) * float(gl)
        seen = ema >= 0
        # STATIONARY and SCALE-INVARIANT: softmax over each group's loss RELATIVE to the
        # current mean, not its absolute value. Absolute-scale softmax self-annihilates
        # as training converges — every group's loss shrinks toward zero, so the gaps
        # between them shrink too and q goes uniform: the tau=0.3 run fell from a 2.8x
        # weight ratio at step 700 to 1.18x by 6300, i.e. ERM with rounding error, while
        # the *relative* gap (hausa 2x zulu) was undiminished. Dividing by the mean makes
        # the tilt depend on how much harder a group is, which is the quantity that
        # actually persists. Centring by the max is numerical only.
        rel = ema[seen] / ema[seen].mean().clamp(min=1e-8)
        z = rel / max(eta, 1e-8)
        w_seen = torch.softmax(z - z.max(), dim=0)
        q.zero_()
        q[seen] = w_seen
        q.clamp_(min=Q_FLOOR)
        q /= q.sum()
    w = torch.stack([q[g] for g in per_group])
    return sum(wi * gl for wi, gl in zip(w / w.sum(), per_group.values())), q


def build_optimizer(model, lr):
    try:
        import bitsandbytes as bnb
        print("optimizer: bitsandbytes AdamW8bit")
        return bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.0)
    except Exception as e:
        print(f"optimizer: torch AdamW (bitsandbytes unavailable: {type(e).__name__}). "
              f"If this OOMs, `pip install bitsandbytes` and rerun.")
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--arm", choices=["erm", "dro"], default="erm")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--dro-eta", type=float, default=0.3,
                    help="DRO group-weight temperature tau: q_g ∝ exp(ema_loss_g / tau). "
                         "Lower = sharper tilt toward the hardest group.")
    ap.add_argument("--save-every", type=int, default=100,
                    help="resume checkpoint interval; the 2026-08-12 run died at step 150, "
                         "before its first save at 250, and lost everything")
    ap.add_argument("--snapshot-every", type=int, default=5000,
                    help="keep a standalone model snapshot this often (~1 GB each), so "
                         "checkpoint selection is possible without retraining")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="20 steps, tiny batch, report VRAM")
    args = ap.parse_args()

    if args.smoke:
        args.steps, args.batch, args.grad_accum, args.save_every = 20, 2, 1, 1000
        args.warmup = 2
        # Never share an output directory with a real run: a 20-step `final/` left
        # behind is indistinguishable from a trained model once the log scrolls away.
        args.out = args.out or str(ROOT / "results" / "ft" / "_smoke")
    out = Path(args.out or (ROOT / "results" / "ft" / args.arm))
    out.mkdir(parents=True, exist_ok=True)

    # A run that dies before its first checkpoint leaves a stale `final/` from whatever
    # ran here previously. Say so loudly rather than letting it be evaluated later.
    if (out / "final").exists() and not args.resume:
        print(f"WARNING: {out / 'final'} already exists and will be overwritten only if "
              f"this run completes. Until then it is STALE — do not evaluate it.")
    set_seed(SEED)

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    prep = Path(args.data_dir) / "prepared"
    groups = sorted(json.loads((ROOT / "groups_afrispeech.json").read_text())["groups"])
    pins = load_pins()
    rev = pins["models"].get(BASE)

    processor = WhisperProcessor.from_pretrained(BASE, revision=rev, language="en",
                                                 task="transcribe")
    train = AfriSpeech(prep, "train", processor, groups)
    print(f"arm={args.arm}  train={len(train)} utt  {train.hours():.2f} h  groups={groups}")
    if train.dropped["long_audio"] or train.dropped["long_labels"]:
        print(f"  filtered: {train.dropped['long_audio']} over {MAX_AUDIO_S:.0f}s audio, "
              f"{train.dropped['long_labels']} over {MAX_LABEL_TOKENS} label tokens; "
              f"by group {train.dropped['by_group']}")

    model = WhisperForConditionalGeneration.from_pretrained(BASE, revision=rev).cuda()
    model.config.forced_decoder_ids = None
    # Non-reentrant checkpointing: Whisper's input_features carry no grad, and the
    # reentrant implementation loses the graph on backward ("backward through the
    # graph a second time"). Also needs use_cache off, set below.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()

    opt = build_optimizer(model, args.lr)
    scaler = torch.amp.GradScaler("cuda")
    # pct_start must stay in (0, 1): a short run with the default warmup would otherwise
    # ask for more warmup steps than the run has.
    pct = min(0.3, max(args.warmup / max(args.steps, 1), 1e-3))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=pct, anneal_strategy="cos")
    q = torch.ones(len(groups), device="cuda") / len(groups)
    # -1 marks "never seen"; groups only enter the q update once observed.
    ema = torch.full((len(groups),), -1.0, device="cuda")

    start = 0
    ckpt_path = out / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        sched.load_state_dict(ck["sched"])
        q = ck["q"].cuda()
        ema = ck.get("ema", torch.full((len(groups),), -1.0)).cuda()
        start = ck["step"]
        print(f"resumed from step {start}")

    loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=2,
                        collate_fn=lambda b: collate(b, processor.tokenizer.pad_token_id),
                        pin_memory=True, drop_last=True)

    log = (out / "train_log.jsonl").open("a", encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps({
        "arm": args.arm, "base": BASE, "base_revision": rev, "seed": SEED,
        "steps": args.steps, "batch": args.batch, "grad_accum": args.grad_accum,
        "effective_batch": args.batch * args.grad_accum, "lr": args.lr,
        "dro_eta": args.dro_eta if args.arm == "dro" else None,
        "groups": groups, "n_train": len(train), "train_hours": round(train.hours(), 3),
        "filters": {"max_audio_s": MAX_AUDIO_S, "max_label_tokens": MAX_LABEL_TOKENS,
                    "dropped": train.dropped},
        "provenance": run_provenance(),
    }, indent=2))

    step, t0, run_loss, since_log = start, time.time(), 0.0, 0
    gloss = defaultdict(list)
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            feats = batch["input_features"].cuda(non_blocking=True)
            labels = batch["labels"].cuda(non_blocking=True)
            gs = batch["groups"].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_features=feats, labels=labels).logits
            losses = per_sample_loss(logits, labels)
            loss, q = reduce_loss(losses, gs, args.arm, q, args.dro_eta, len(groups), ema)
            scaler.scale(loss / args.grad_accum).backward()
            run_loss += float(loss.detach())
            since_log += 1
            for g, ls in zip(gs.tolist(), losses.tolist()):
                gloss[groups[g]].append(ls)

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            step += 1

            if step % 25 == 0 or args.smoke:
                vram = torch.cuda.max_memory_reserved() / 2**30
                rec = {"step": step, "loss": round(run_loss / max(since_log, 1), 4),
                       "lr": round(sched.get_last_lr()[0], 8), "vram_gb": round(vram, 2),
                       "sec_per_step": round((time.time() - t0) / max(step - start, 1), 3),
                       "group_loss": {g: round(float(np.mean(v)), 4)
                                      for g, v in gloss.items() if v},
                       "q": {g: round(float(q[i]), 4) for i, g in enumerate(groups)},
                       "ema_loss": {g: round(float(ema[i]), 4) for i, g in enumerate(groups)
                                    if float(ema[i]) >= 0}}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                run_loss, gloss, since_log = 0.0, defaultdict(list), 0

            if step % args.save_every == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "scaler": scaler.state_dict(), "sched": sched.state_dict(),
                            "q": q.cpu(), "ema": ema.cpu(), "step": step}, out / "checkpoint.tmp")
                os.replace(out / "checkpoint.tmp", ckpt_path)   # atomic: survives a crash mid-save

            if args.snapshot_every and step % args.snapshot_every == 0:
                # Model-only, no optimizer state: enough to evaluate this point later
                # without retraining, at ~1 GB rather than ~2.5 GB.
                model.save_pretrained(out / f"step{step}")
                print(f"snapshot: {out / f'step{step}'}", flush=True)

    model.save_pretrained(out / "final")
    processor.save_pretrained(out / "final")
    print(f"\ndone: {out / 'final'}  ({(time.time() - t0) / 3600:.2f} h)")
    if args.smoke:
        print(f"peak VRAM {torch.cuda.max_memory_reserved() / 2**30:.2f} GB — "
              f"if under ~7 GB, the overnight run at --batch 4 should hold.")


if __name__ == "__main__":
    main()
