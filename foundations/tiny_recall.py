"""Exp 1C: why do long prompts hurt? A tiny GPT whose every part we control.

Task: synthetic key -> value lookup (multi-query associative recall, as in Zoology,
Arora et al. 2023).

    k k k v  k k k v  ...  k k k v | Q k k k v  Q k k k v  ...  Q k k k  ->  v
    <------------ N facts -------->  <---------- 8 queries ----------->

A key is 3 symbols, a value is 1 token. The model reads N facts, then 8 queries (8
distinct keys from the facts); after each query key it must output that key's value.
Earlier queries are followed by their answer, so one sequence gives 8 training signals
(with a single query the model barely learns: tried, loss stays near chance). Accuracy
and attention are measured on the **final** query only. The fact that answers it is the
"needle"; the other facts are distractors. Like a RAG prompt, the answer is always in
context: only the amount of distracting text (N) changes.

  related distractors:   every distractor key shares its first symbol with the query key
  unrelated distractors: random keys

Training uses N = 8 to 64 facts, drawn per batch (half related, half unrelated): short
sequences let the model discover the lookup mechanism, which it fails to find when
trained on N = 64 alone (loss stays at chance). Testing uses N = 16 to 512, so lengths
above 64 test length generalization.

Variants (one change at a time):
  --pos rope      rotary position embedding (as in Qwen, Llama)       [baseline]
  --pos learned   learned absolute positions (as in GPT-2). Positions beyond the training
                  length are never trained.
  --pos none      no position information except the causal mask
  --pos rope --ssmax   rope + scalable softmax: attention logits are multiplied by a
                  learned s * log(n), n = number of positions attended to, so attention
                  can stay sharp as n grows (Nakanishi, 2025)

Inference-time intervention, applied to every trained model at evaluation:
  logn      multiply attention logits by max(1, log n / log n_train). Identical to the
            standard model inside the training length; sharper beyond it. Same weights,
            so any change in accuracy is *caused* by the attention temperature.

We record accuracy and the attention share on the needle fact (from the final position,
per layer). The attention-dilution hypothesis predicts that share falls as N grows, and
that sharpening attention (logn / ssmax) slows that fall and the accuracy loss.

  python foundations/tiny_recall.py --pos rope --seed 0      # one run, a few minutes on a T4
  python foundations/tiny_recall.py --sweep                  # 4 variants x 3 seeds
Results: results/tiny/<variant>_s<seed>.json, summarized by scripts/analyze.py.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

N_SYM, N_VAL, KEY_LEN = 32, 64, 3
QUERY = N_SYM + N_VAL
VOCAB = QUERY + 1
FACT = KEY_LEN + 1
N_TRAIN = 64
N_QUERIES = 8
TEST_N = (16, 32, 64, 128, 256, 512)
QUERY_BLOCK = (1 + FACT) * N_QUERIES - 1          # "Q k k k v" x 8, minus the final answer
MAX_LEN = FACT * max(TEST_N) + QUERY_BLOCK
T_TRAIN = FACT * N_TRAIN + QUERY_BLOCK

# --------------------------------------------------------------------------- data


def make_batch(batch: int, n_facts: int, kind: str, gen: torch.Generator, device="cpu"):
    """Returns tokens [B, T], answers [B, Q], needle indices [B, Q], and the positions
    [Q] whose next-token prediction is each query's answer (the last key symbol).

    kind: 'related' | 'unrelated' | 'mixed' (half of the batch each).
    Keys inside one sequence are distinct; the Q queried keys are distinct.
    """
    if kind == "mixed":
        a = make_batch(batch // 2, n_facts, "related", gen)
        b = make_batch(batch - batch // 2, n_facts, "unrelated", gen)
        return (*(torch.cat([x, y]).to(device) for x, y in zip(a[:3], b[:3])), a[3].to(device))
    keys = torch.empty(batch, n_facts, KEY_LEN, dtype=torch.long)
    for i in range(batch):
        # sample distinct keys as integers in [0, N_SYM^3), then split into symbols
        if kind == "related":
            first = torch.randint(N_SYM, (1,), generator=gen).item()
            rest = torch.randperm(N_SYM ** (KEY_LEN - 1), generator=gen)[:n_facts]
            code = first * N_SYM ** (KEY_LEN - 1) + rest
        else:
            code = torch.randperm(N_SYM ** KEY_LEN, generator=gen)[:n_facts]
        for j in reversed(range(KEY_LEN)):
            keys[i, :, j] = code % N_SYM
            code = code // N_SYM
    vals = torch.randint(N_VAL, (batch, n_facts), generator=gen) + N_SYM
    needle = torch.stack([torch.randperm(n_facts, generator=gen)[:N_QUERIES] for _ in range(batch)])
    rows = torch.arange(batch)[:, None]
    qkeys, answers = keys[rows, needle], vals[rows, needle]                 # [B, Q, 3], [B, Q]
    queries = torch.cat([torch.full((batch, N_QUERIES, 1), QUERY), qkeys, answers[..., None]], -1)
    facts = torch.cat([keys, vals[..., None]], dim=-1).reshape(batch, -1)
    tokens = torch.cat([facts, queries.reshape(batch, -1)[:, :-1]], dim=1)  # drop the final answer
    qpos = FACT * n_facts + (1 + FACT) * torch.arange(N_QUERIES) + KEY_LEN
    return tokens.to(device), answers.to(device), needle.to(device), qpos.to(device)


# --------------------------------------------------------------------------- model


def rope(x: torch.Tensor) -> torch.Tensor:
    """Rotary embedding on [B, H, T, D]: rotate each pair of dimensions by an angle that
    grows with position, so q·k depends on the *relative* distance between tokens."""
    T, D = x.shape[-2], x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device=x.device).float() / D))
    ang = torch.arange(T, device=x.device).float()[:, None] * inv[None]  # [T, D/2]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d, n_head, pos, ssmax):
        super().__init__()
        self.h, self.hd, self.pos = n_head, d // n_head, pos
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        # scalable softmax: one learned s per head, initialised so s * log(T_TRAIN) = 1
        self.s = nn.Parameter(torch.full((n_head,), 1 / math.log(T_TRAIN))) if ssmax else None

    def forward(self, x, logn: bool, keep_last_row: bool):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        if self.pos == "rope":
            q, k = rope(q), rope(k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)        # [B, H, T, T]
        n = torch.arange(1, T + 1, device=x.device).float()         # positions attended to
        if self.s is not None:
            att = att * (self.s[:, None, None] * n.log()[:, None]).clamp(min=1e-2)
        if logn:
            att = att * (n.log() / math.log(T_TRAIN)).clamp(min=1.0)[:, None]
        att = att.masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
        att = att.softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y), (att[:, :, -1, :].mean(1) if keep_last_row else None)  # [B, T]


class Block(nn.Module):
    def __init__(self, d, n_head, pos, ssmax):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, n_head, pos, ssmax)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, logn, keep):
        a, row = self.attn(self.ln1(x), logn, keep)
        x = x + a
        return x + self.mlp(self.ln2(x)), row


class TinyGPT(nn.Module):
    def __init__(self, pos="rope", ssmax=False, d=128, n_head=4, n_layer=4):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.pos_emb = nn.Embedding(MAX_LEN, d) if pos == "learned" else None
        self.blocks = nn.ModuleList(Block(d, n_head, pos, ssmax) for _ in range(n_layer))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, idx, logn=False, keep_attention=False):
        x = self.tok(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(idx.shape[1], device=idx.device))
        rows = []
        for b in self.blocks:
            x, row = b(x, logn, keep_attention)
            rows.append(row)
        return self.head(self.ln(x)), rows  # logits [B, T, VOCAB]


# --------------------------------------------------------------------------- train / eval


def train(model, steps, device, seed, batch=64, lr=1e-3, max_facts=N_TRAIN, min_facts=N_QUERIES):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    curve = []
    for step in range(steps):
        n = int(torch.randint(min_facts, max_facts + 1, (1,), generator=gen))
        x, y, _, qpos = make_batch(batch, n, "mixed", gen, device)
        logits = model(x)[0][:, qpos]                                  # [B, Q, VOCAB]
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == steps - 1:
            acc = (logits.argmax(-1) == y).float().mean().item()
            curve.append({"step": step, "loss": round(loss.item(), 4), "acc": round(acc, 4)})
            print(f"  step {step:5d} loss {loss.item():.3f} acc {acc:.3f}", flush=True)
    return curve


@torch.no_grad()
def evaluate(model, device, n_eval=512, batch=16, seed=10_000):
    model.eval()
    out = {}
    for kind in ("related", "unrelated"):
        out[kind] = {}
        for n in TEST_N:
            out[kind][n] = {}
            for logn in (False, True):
                gen = torch.Generator().manual_seed(seed + n)  # same test set for both modes
                correct, share = 0, None
                for _ in range(n_eval // batch):
                    x, y, needle, _ = make_batch(batch, n, kind, gen, device)
                    y, needle = y[:, -1], needle[:, -1]                    # final query only
                    logits, rows = model(x, logn=logn, keep_attention=True)
                    correct += (logits[:, -1].argmax(-1) == y).sum().item()
                    # attention from the final position onto the needle fact's 4 tokens
                    idx = (needle[:, None] * FACT + torch.arange(FACT, device=device)[None])  # [B, 4]
                    s = torch.stack([r.gather(1, idx).sum(1) for r in rows], 1).sum(0)        # [layers]
                    share = s if share is None else share + s
                out[kind][n]["logn" if logn else "standard"] = {
                    "acc": correct / (n_eval // batch * batch),
                    "needle_share": (share / (n_eval // batch * batch)).tolist(),
                    "uniform_share": FACT / (FACT * n + QUERY_BLOCK),
                }
            r = out[kind][n]
            print(f"  {kind:9s} N={n:4d} acc {r['standard']['acc']:.3f} (logn {r['logn']['acc']:.3f})  "
                  f"needle share, last layer {r['standard']['needle_share'][-1]:.3f}", flush=True)
    return out


VARIANTS = {"rope": ("rope", False), "learned": ("learned", False), "none": ("none", False),
            "rope_ssmax": ("rope", True)}


def run(variant, seed, steps, device, out_dir=Path("results/tiny")):
    pos, ssmax = VARIANTS[variant]
    torch.manual_seed(seed)
    model = TinyGPT(pos=pos, ssmax=ssmax).to(device)
    print(f"{variant} seed {seed}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    t0 = time.time()
    curve = train(model, steps, device, seed)
    if curve[-1]["acc"] < 0.95:
        print(f"  WARNING: final training accuracy {curve[-1]['acc']:.2f} < 0.95; the model has not "
              "learned the task at the training length. Rerun with more --steps before comparing variants.")
    res = {"variant": variant, "seed": seed, "steps": steps, "train_curve": curve,
           "eval": evaluate(model, device), "minutes": round((time.time() - t0) / 60, 2)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{variant}_s{seed}.json").write_text(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", choices=["rope", "learned", "none"], default="rope")
    ap.add_argument("--ssmax", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--sweep", action="store_true", help="all 4 variants x seeds 0, 1, 2")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.sweep:
        for variant in VARIANTS:
            for seed in (0, 1, 2):
                if Path(f"results/tiny/{variant}_s{seed}.json").exists():
                    print(f"{variant} seed {seed}: done, skipping")
                    continue
                run(variant, seed, args.steps, device)
    else:
        variant = "rope_ssmax" if args.ssmax else args.pos
        if args.ssmax and args.pos != "rope":
            raise SystemExit("--ssmax is defined on top of --pos rope")
        run(variant, args.seed, args.steps, device)


if __name__ == "__main__":
    main()
