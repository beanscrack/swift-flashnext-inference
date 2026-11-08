"""P3b: self-distillation fine-tune of the MTP draft head (train split only).

Weights: bf16 model (forward/backward, eval path), fp32 master copies + AdamW state.
Loss: per completion anchor e in [P-1, T-6], per step j=0..3, CE over the 49,152 draft-vocab
rows against target_argmax[e+1+j] (labels outside the draft vocab ignored); per-sequence mean over
anchors per step, then equal weight over steps; sequences averaged over 8-sequence accumulation groups.
"""
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3b")
sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3")
from common import P3B, ORIG, ORIG_RT, load_base, sha, eval_round_end, export_bf16  # noqa: E402
from ttt import ttt_forward, K_STEPS  # noqa: E402
from data import load_gen, load_sample  # noqa: E402
from moe_fast import set_fast  # noqa: E402  (training forward only; eval uses the original MoE code)
from safetensors import safe_open  # noqa: E402

LOGS = f"{P3B}/logs"
EVAL = f"{P3B}/eval"
CKPT = f"{P3B}/ckpt"
PEAK_LR = 2e-5
WARMUP = 50
MIN_FRAC = 0.1
BETAS = (0.9, 0.95)
WD = 0.0
CLIP = 1.0
ACC = 8
EPOCHS = 2
EVAL_EVERY = 100
SEED = 0
ORIG_ACC3 = ORIG_RT["acc"][3]
ORIG_TPC = ORIG_RT["tokens_per_cycle"]


def main():
    for d in (LOGS, EVAL, CKPT):
        os.makedirs(d, exist_ok=True)
    torch.manual_seed(SEED)
    logf = open(f"{LOGS}/train_steps.jsonl", "w")
    evalf = open(f"{LOGS}/evals.jsonl", "w")

    def say(msg):
        print(msg, flush=True)

    model, lm_head, lm_rows, vocab = load_base()
    set_fast(True)
    model.requires_grad_(False)
    named = [(n, p) for n, p in model.named_parameters() if n != "embed_tokens.weight"]
    assert len(named) == 31, len(named)
    frozen_before = {"embed_tokens": sha(model.embed_tokens.weight), "lm_head": sha(lm_head)}
    with safe_open(ORIG, framework="pt", device="cpu") as fo:
        assert {"mtp." + n for n, _ in named} == set(fo.keys())
        for n, p in named:
            assert torch.equal(p.detach().cpu(), fo.get_tensor("mtp." + n)), n
    say("start weights: model-tail mtp.* bitwise equal to mtp_original (31/31)")
    for _, p in named:
        p.requires_grad_(True)
    masters = [p.detach().float().clone() for _, p in named]  # fp32 master weights
    opt = torch.optim.AdamW(masters, lr=PEAK_LR, betas=BETAS, weight_decay=WD, foreach=False)

    train_all = load_gen("train")
    held = load_gen("heldout")
    assert len(train_all) == 3982 and len(held) == 204, (len(train_all), len(held))
    held_ids = {r["id"] for r in held}
    assert not (held_ids & {r["id"] for r in train_all}), "heldout/train overlap"
    train = [r for r in train_all if r["C"] >= 5]  # C<5: no completion anchor (P-1 > T-6)
    n_train = len(train)
    steps_per_epoch = math.ceil(n_train / ACC)
    total_steps = EPOCHS * steps_per_epoch
    say(f"train samples {len(train_all)} (used {n_train}; {len(train_all)-n_train} with C<5 have no anchors), "
        f"heldout {len(held)}, steps/epoch {steps_per_epoch}, total {total_steps}")

    def lr_at(s):
        if s < WARMUP:
            return PEAK_LR * (s + 1) / WARMUP
        prog = (s - WARMUP) / max(1, total_steps - WARMUP)
        return PEAK_LR * (MIN_FRAC + (1 - MIN_FRAC) * 0.5 * (1 + math.cos(math.pi * prog)))

    inv = torch.full((lm_head.shape[0],), -1, dtype=torch.long, device="cuda")
    inv[vocab] = torch.arange(vocab.numel(), device="cuda")

    stats = {"heldout_samples_in_gradient": 0, "train_samples_in_gradient": 0,
             "skipped_no_valid_label": 0}

    def sample_loss(rec):
        assert rec["split"] == "train" and rec["id"] not in held_ids, rec["id"]
        hid, x, labels, T, P, C = load_sample(rec["id"])
        assert P == rec["P"] and C == rec["C"]
        anchors = torch.arange(P - 1, T - 5, device="cuda")  # e in [P-1, T-6]
        S = ttt_forward(model, hid, x, anchors=anchors)  # [4, N, H], bf16, differentiable
        step_losses = []
        for j in range(K_STEPS):
            tgt = inv[labels[anchors + 1 + j]]  # draft-vocab index, -1 if outside the draft vocab
            valid = tgt >= 0
            if int(valid.sum()) == 0:
                continue
            logits = F.linear(S[j], lm_rows).float()  # [N, 49152]
            step_losses.append(F.cross_entropy(logits[valid], tgt[valid]))
        if not step_losses:
            return None, T
        return torch.stack(step_losses).mean(), T  # equal weight over steps

    evals = []
    best = {"tpc": ORIG_TPC, "step": None}

    def do_eval(step, epoch):
        t0 = time.time()
        set_fast(False)  # eval path = original simulate.py/ttt/MoE code
        try:
            d = eval_round_end(model, lm_rows, vocab, f"s{step:04d}", EVAL)
        finally:
            set_fast(True)
        re = d["round_end"]
        ev = {"step": step, "epoch": epoch, "acc": re["acc"], "tokens_per_cycle": re["tokens_per_cycle"],
              "n_rounds": re["n"], "round_end_by_mode": d["round_end_by_mode"],
              "overall": d["overall"], "by_mode": d["by_mode"], "eval_seconds": time.time() - t0}
        evals.append(ev)
        evalf.write(json.dumps(ev) + "\n")
        evalf.flush()
        say(f"EVAL step {step} epoch {epoch}: acc {['%.4f' % a for a in re['acc']]} "
            f"tokens/cycle {re['tokens_per_cycle']:.4f} ({ev['eval_seconds']:.0f}s)")
        if step > 0:
            export_bf16(named, f"{CKPT}/latest.safetensors")
            if re["tokens_per_cycle"] > best["tpc"]:
                export_bf16(named, f"{CKPT}/best.safetensors")
                best.update(tpc=re["tokens_per_cycle"], step=step, acc=re["acc"], epoch=epoch)
                with open(f"{CKPT}/best_meta.json", "w") as f:
                    json.dump(best, f, indent=1)
                say(f"  saved best (step {step})")
        return ev

    # ---- STOP RULE 1: original weights through the training-side eval path ----
    ev0 = do_eval(0, 0)
    dev = max(abs(a - b) for a, b in zip(ev0["acc"], ORIG_RT["acc"]))
    say(f"original check: max |acc - 0.812/0.655/0.531/0.431| = {dev:.5f}")
    if dev > 0.005:
        say("STOP RULE 1 VIOLATED: eval path does not reproduce original numbers")
        json.dump({"stop": "rule1", "eval0": ev0}, open(f"{LOGS}/summary.json", "w"), indent=1)
        sys.exit(3)

    train_t, train_tokens, peak_alloc = 0.0, 0, 0.0
    bad_streak = 0
    stop_reason = None
    rng = random.Random(SEED)
    step = 0
    skipped = []
    for epoch in range(1, EPOCHS + 1):
        order = list(range(n_train))
        rng.shuffle(order)
        for g in range(steps_per_epoch):
            group = [train[i] for i in order[g * ACC:(g + 1) * ACC]]
            n_micro = len(group)
            lr = lr_at(step)
            for pg in opt.param_groups:
                pg["lr"] = lr
            t0 = time.time()
            loss_sum, n_used, tok = 0.0, 0, 0
            for rec in group:
                loss, T = sample_loss(rec)
                tok += T
                if loss is None:
                    skipped.append(rec["id"])
                    continue
                (loss / n_micro).backward()
                loss_sum += float(loss.detach())
                n_used += 1
                stats["train_samples_in_gradient"] += 1
                for m, (_, p) in zip(masters, named):
                    if p.grad is not None:
                        if m.grad is None:
                            m.grad = p.grad.float()
                        else:
                            m.grad.add_(p.grad)
                        p.grad = None
            gn = float(torch.nn.utils.clip_grad_norm_(masters, CLIP))
            opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for m, (_, p) in zip(masters, named):
                    p.copy_(m)  # fp32 master -> bf16 compute weights
            torch.cuda.synchronize()
            dt = time.time() - t0
            step += 1
            train_t += dt
            train_tokens += tok
            peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated() / 2**30)
            rec_log = {"step": step, "epoch": epoch, "loss": loss_sum / max(n_used, 1), "lr": lr,
                       "grad_norm": gn, "sec": dt, "tokens": tok, "n_used": n_used}
            logf.write(json.dumps(rec_log) + "\n")
            logf.flush()
            if step % 10 == 0:
                say(f"step {step}/{total_steps} ep {epoch} loss {rec_log['loss']:.4f} lr {lr:.2e} "
                    f"gn {gn:.3f} {dt:.2f}s peakGiB {peak_alloc:.1f}")
            if step % EVAL_EVERY == 0 or g == steps_per_epoch - 1:
                ev = do_eval(step, epoch)
                if g == steps_per_epoch - 1 and epoch == 1:
                    # ---- STOP RULE 2: after epoch 1, acc_3 must improve by >= 0.03 over 0.431
                    if ev["acc"][3] - ORIG_ACC3 < 0.03:
                        stop_reason = (f"stop rule 2: epoch-1 acc_3 {ev['acc'][3]:.4f} "
                                       f"improved by {ev['acc'][3] - ORIG_ACC3:.4f} < 0.03")
                        break
                # ---- STOP RULE 3: 3 consecutive evals with tokens/cycle below the original
                bad_streak = bad_streak + 1 if ev["tokens_per_cycle"] < ORIG_TPC else 0
                if bad_streak >= 3:
                    stop_reason = "stop rule 3: 3 consecutive evals below original tokens/cycle"
                    break
        if stop_reason:
            break

    if stop_reason:
        say("STOP: " + stop_reason)
    frozen_after = {"embed_tokens": sha(model.embed_tokens.weight), "lm_head": sha(lm_head)}
    summary = {
        "stop_reason": stop_reason, "steps_done": step, "total_steps": total_steps,
        "n_train_used": n_train, "n_heldout": len(held), "heldout_ids_in_gradient": stats["heldout_samples_in_gradient"],
        "train_samples_in_gradient_total": stats["train_samples_in_gradient"],
        "skipped_no_valid_label": skipped,
        "frozen_before": frozen_before, "frozen_after": frozen_after,
        "frozen_identical": frozen_before == frozen_after,
        "best": best, "evals": evals,
        "train_seconds": train_t, "train_tokens": train_tokens,
        "tokens_per_s": train_tokens / max(train_t, 1e-9),
        "peak_allocated_GiB": peak_alloc,
        "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
        "hparams": {"peak_lr": PEAK_LR, "warmup": WARMUP, "min_frac": MIN_FRAC, "betas": BETAS, "wd": WD,
                    "clip": CLIP, "acc": ACC, "epochs": EPOCHS, "eval_every": EVAL_EVERY, "seed": SEED},
    }
    with open(f"{LOGS}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    say(f"DONE steps {step} best {best} frozen_identical {summary['frozen_identical']}")


if __name__ == "__main__":
    main()
