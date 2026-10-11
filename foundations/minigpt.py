"""A small GPT, written from scratch. Learning exercise for the attention analysis in H5.

Follow Karpathy's "Let's build GPT: from scratch, in code, spelled out"
(https://www.youtube.com/watch?v=kCc8FmEb1nY) and fill in each TODO yourself. The
data loading and training loop are done; the model is yours to write.

Run on a Colab T4:  python foundations/minigpt.py
Done when: validation loss on TinyShakespeare goes below ~1.6 and samples look like
(bad) Shakespeare.

Questions to be able to answer afterwards (they come up in H5 and in interviews):
  1. Why is attention scaled by 1/sqrt(head_dim)?
  2. What does the causal mask do, and where exactly is it applied?
  3. Why multiple heads instead of one big head?
  4. What do residual connections and LayerNorm each fix?
  5. In H5 we read attention rows of the *answer* tokens. Why those rows and not the
     columns?
"""
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters (fits a T4 in a few minutes)
BATCH, BLOCK, STEPS, LR = 64, 256, 5000, 3e-4
N_EMBD, N_HEAD, N_LAYER, DROPOUT = 384, 6, 6, 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# ------------------------------------------------------------------ data (done)
path = Path("data/raw/tinyshakespeare.txt")
if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", path)
text = path.read_text()
chars = sorted(set(text))
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK, (BATCH,))
    x = torch.stack([d[i:i + BLOCK] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ------------------------------------------------------------------ model (yours)
class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, head_size):
        super().__init__()
        # TODO: key, query, value projections (nn.Linear(N_EMBD, head_size, bias=False))
        # TODO: register_buffer a lower-triangular mask of shape (BLOCK, BLOCK)
        # TODO: dropout on the attention weights
        raise NotImplementedError

    def forward(self, x):
        # TODO: q @ k^T scaled by head_size**-0.5, mask future positions with -inf,
        #       softmax over the last dim, dropout, then weights @ v
        raise NotImplementedError


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        # TODO: n_head Heads, an output projection back to N_EMBD, dropout
        raise NotImplementedError

    def forward(self, x):
        # TODO: concatenate head outputs on the last dim, project, dropout
        raise NotImplementedError


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Linear(N_EMBD, 4*N_EMBD) -> GELU/ReLU -> Linear(4*N_EMBD, N_EMBD) -> Dropout
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: LayerNorm -> attention, LayerNorm -> feed-forward (pre-norm)
        raise NotImplementedError

    def forward(self, x):
        # TODO: x = x + attn(ln1(x)); x = x + ffwd(ln2(x))
        raise NotImplementedError


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: token embedding (VOCAB, N_EMBD), position embedding (BLOCK, N_EMBD),
        #       N_LAYER Blocks, final LayerNorm, lm_head Linear(N_EMBD, VOCAB)
        raise NotImplementedError

    def forward(self, idx, targets=None):
        # TODO: embeddings -> blocks -> ln_f -> logits; if targets: cross-entropy loss
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # TODO: crop to last BLOCK tokens, take last-step logits, softmax, sample, append
        raise NotImplementedError


# ------------------------------------------------------------------ training (done)
@torch.no_grad()
def estimate_loss(model, iters=200):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(iters)
        for k in range(iters):
            _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


if __name__ == "__main__":
    model = GPT().to(DEVICE)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for step in range(STEPS):
        if step % 500 == 0 or step == STEPS - 1:
            print(step, estimate_loss(model))
        _, loss = model(*get_batch("train"))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device=DEVICE), 500)[0].tolist()))
