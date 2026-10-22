"""Exp 2: how much attention do the answer tokens put on each passage?

Method
------
1. Rebuild the exact QA prompt the generator saw and append the model's own greedy
   answer (teacher forcing: one forward pass over prompt + answer).
2. Map every passage to its token span using the tokenizer's character offsets.
3. In every layer, take the attention rows of the positions that *predict* the answer
   tokens, average over heads and answer tokens, and sum the mass inside each passage.
4. Normalize per layer so the passage shares sum to 1 (mass on the instruction, the
   question and special tokens is dropped). Result: array [n_layers, n_passages].

Why not ``output_attentions=True``? Eager attention materializes a [heads, L, L] matrix
per layer: at L = 28K tokens that is ~44 GB for one layer. We only need the few rows
belonging to the answer, so a pre-hook on every attention module recomputes just those
rows from the layer's own inputs (q for the answer rows, k for all positions, RoPE,
causal mask, softmax). The model itself runs with SDPA, and the forward pass skips the
LM head (its 28K x 152K logits would be another ~17 GB). ``check_against_eager``
verifies the recomputed rows against real eager attention on a short input.

Caveat (stated in HYPOTHESES.md): attention weights are not a faithful explanation of
model behavior (Jain & Wallace, 2019). Exp 2 measures association, not cause.
"""
from __future__ import annotations

import numpy as np

from .pipelines import format_passage, format_passages, qa_body


class AttentionProbe:
    def __init__(self, model, tokenizer, device: str = "cuda"):
        import torch

        self.torch = torch
        self.model = model.eval()
        self.tok = tokenizer
        self.device = device
        self.layers = model.model.layers
        self._rows: tuple[int, int] | None = None
        self._per_layer: dict[int, object] = {}
        for li, layer in enumerate(self.layers):
            layer.self_attn.register_forward_pre_hook(self._make_hook(li), with_kwargs=True)

    @classmethod
    def from_pretrained(cls, name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device)
        return cls(model, AutoTokenizer.from_pretrained(name), device)

    # ------------------------------------------------------------------ attention rows

    def _make_hook(self, li: int):
        def hook(module, args, kwargs):
            if self._rows is None:
                return None
            hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            pe = kwargs["position_embeddings"] if "position_embeddings" in kwargs else args[1]
            self._per_layer[li] = self._answer_rows(module, hs, pe)
            return None
        return hook

    def _answer_rows(self, module, hidden_states, position_embeddings):
        """Mean (over heads and answer rows) attention distribution over all positions."""
        torch = self.torch
        from transformers.models.qwen2.modeling_qwen2 import rotate_half

        r0, r1 = self._rows
        h = hidden_states[0]                                        # [T, D]
        T = h.shape[0]
        hd = module.head_dim
        q = module.q_proj(h[r0:r1]).view(r1 - r0, -1, hd).transpose(0, 1)   # [H, m, hd]
        k = module.k_proj(h).view(T, -1, hd).transpose(0, 1)                # [KV, T, hd]
        cos, sin = position_embeddings                              # [1, T, hd] each
        cos, sin = cos[0], sin[0]
        q = q * cos[r0:r1] + rotate_half(q) * sin[r0:r1]
        k = k * cos + rotate_half(k) * sin
        k = k.repeat_interleave(q.shape[0] // k.shape[0], dim=0)    # grouped-query attention
        scores = torch.matmul(q.float(), k.float().transpose(1, 2)) * module.scaling  # [H, m, T]
        rows = torch.arange(r0, r1, device=h.device)[:, None]
        cols = torch.arange(T, device=h.device)[None, :]
        scores = scores.masked_fill(cols > rows, float("-inf"))     # causal mask
        return scores.softmax(dim=-1).mean(dim=(0, 1)).cpu()        # [T]

    def answer_attention(self, input_ids: list[int], rows: tuple[int, int]) -> np.ndarray:
        """[n_layers, T]: per layer, attention of rows[0]:rows[1] averaged over heads/rows."""
        torch = self.torch
        self._rows, self._per_layer = rows, {}
        try:
            with torch.no_grad():
                self.model.model(input_ids=torch.tensor([input_ids], device=self.device), use_cache=False)
        finally:
            self._rows = None
        if len(self._per_layer) != len(self.layers):
            raise RuntimeError(f"captured {len(self._per_layer)} of {len(self.layers)} layers")
        return np.stack([self._per_layer[i].numpy() for i in range(len(self.layers))])

    def check_against_eager(self, input_ids: list[int], rows: tuple[int, int], atol: float = 2e-2) -> float:
        """Max |probe - eager| over layers and positions. Raises if above ``atol``.

        Both are computed in the same eager forward pass, from the same hidden states, so
        the check tests the probe's arithmetic. (Comparing against a separate SDPA forward
        would add the bf16 drift between two forward passes, ~0.03 on Qwen2.5-7B, which is
        not a probe error.) Run on a short input: eager attention is quadratic in memory.
        """
        torch = self.torch
        prev = self.model.config._attn_implementation
        self._set_attn("eager")
        self._rows, self._per_layer = rows, {}
        try:
            with torch.no_grad():
                out = self.model.model(input_ids=torch.tensor([input_ids], device=self.device),
                                       output_attentions=True, use_cache=False)
        finally:
            self._rows = None
            self._set_attn(prev)
        got = np.stack([self._per_layer[i].numpy() for i in range(len(self.layers))])
        ref = np.stack([a[0, :, rows[0]:rows[1], :].float().mean(dim=(0, 1)).cpu().numpy()
                        for a in out.attentions])
        err = float(np.abs(got - ref).max())
        if err > atol:
            raise AssertionError(f"probe differs from eager attention by {err:.4f}")
        return err

    def _set_attn(self, name: str) -> None:
        if hasattr(self.model, "set_attn_implementation"):
            self.model.set_attn_implementation(name)
        else:
            self.model.config._attn_implementation = name

    # ------------------------------------------------------------------ passages

    def encode(self, question: str, passages: list[dict], answer: str):
        """Token ids of prompt + answer, the answer-predicting rows, and passage spans."""
        docs = format_passages(passages)
        prompt = self.tok.apply_chat_template(
            [{"role": "user", "content": qa_body(question, passages)}], tokenize=False, add_generation_prompt=True
        )
        char_spans, cursor = [], prompt.index(docs)
        for i, p in enumerate(passages, start=1):
            piece = format_passage(i, p)
            start = prompt.index(piece, cursor)
            char_spans.append((start, start + len(piece)))
            cursor = start + len(piece)

        enc = self.tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
        offsets = np.asarray(enc["offset_mapping"])
        tok_spans = []
        for s, e in char_spans:
            idx = np.nonzero((offsets[:, 0] < e) & (offsets[:, 1] > s))[0]
            tok_spans.append((int(idx[0]), int(idx[-1]) + 1))

        ans_ids = self.tok(answer, add_special_tokens=False)["input_ids"] or [self.tok.eos_token_id]
        n_prompt = len(enc["input_ids"])
        # The last prompt position predicts answer token 1, and so on.
        rows = (n_prompt - 1, n_prompt - 1 + len(ans_ids))
        return enc["input_ids"] + ans_ids, rows, tok_spans

    def passage_attention(self, question: str, passages: list[dict], answer: str) -> np.ndarray:
        """[n_layers, n_passages]; each row sums to 1 over passages."""
        ids, rows, spans = self.encode(question, passages, answer)
        att = self.answer_attention(ids, rows)
        out = np.stack([att[:, a:b].sum(axis=1) for a, b in spans], axis=1)
        return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


def upper_half_mean(att: np.ndarray) -> np.ndarray:
    """Pre-registered aggregation: mean passage share over the upper half of layers."""
    return att[att.shape[0] // 2:].mean(axis=0)
