"""H5: how much attention do the answer tokens put on each passage?

Method
------
1. Build the same QA prompt the generator saw, then append the model's own greedy answer
   (teacher forcing: one forward pass over prompt + answer).
2. Map every passage to its token span using the tokenizer's character offsets.
3. In every layer, take the attention rows of the answer tokens, average over heads and
   answer tokens, and sum the mass that falls inside each passage span.
4. Normalize per layer so the passage shares sum to 1 (mass on instructions, the
   question and special tokens is dropped). Result: array [n_layers, n_passages].

Memory: eager attention materializes [heads, L, L] per layer. Forward hooks reduce each
layer's weights immediately and return None in their place, so only one layer's matrix
is alive at a time. Keep contexts <= ~4K tokens on a 40GB GPU.

Caveat (stated in HYPOTHESES.md): attention weights are not a faithful explanation of
model behavior (Jain & Wallace, 2019). This is correlational evidence.
"""
from __future__ import annotations

import numpy as np

from .pipelines import QA_INSTRUCTION, format_passages


class AttentionProbe:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to(device).eval()
        self.device = device
        self.layers = self.model.model.layers
        self._answer_slice: slice | None = None
        self._per_layer: list = []
        for layer in self.layers:
            layer.self_attn.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if weights is not None and self._answer_slice is not None:
            # weights: [batch, heads, q_len, k_len] -> mean over heads and answer rows
            rows = weights[0, :, self._answer_slice, :].float().mean(dim=(0, 1))
            self._per_layer.append(rows.cpu())
            return (output[0], None) + tuple(output[2:])
        return output

    def _prompt_and_spans(self, question: str, passages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
        """Chat-formatted prompt plus the character span of each passage inside it."""
        docs = format_passages(passages)
        body = f"{QA_INSTRUCTION} Use the documents below.\n\n{docs}\n\nQuestion: {question}\nAnswer:"
        prompt = self.tok.apply_chat_template(
            [{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True
        )
        spans, cursor = [], prompt.index(docs)
        for i, p in enumerate(passages, start=1):
            piece = f"Document [{i}] (Title: {p['title']}) {p['text']}"
            start = prompt.index(piece, cursor)
            spans.append((start, start + len(piece)))
            cursor = start + len(piece)
        return prompt, spans

    def passage_attention(self, question: str, passages: list[dict], answer: str) -> np.ndarray:
        torch = self.torch
        prompt, char_spans = self._prompt_and_spans(question, passages)
        enc = self.tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
        ans_ids = self.tok(answer, add_special_tokens=False)["input_ids"] or [self.tok.eos_token_id]
        ids = enc["input_ids"] + ans_ids
        n_prompt = len(enc["input_ids"])

        # token span of each passage
        tok_spans = []
        for s, e in char_spans:
            idx = [t for t, (a, b) in enumerate(offsets) if a < e and b > s]
            tok_spans.append((idx[0], idx[-1] + 1))

        # Rows that *predict* answer tokens: the last prompt position predicts answer
        # token 1, ..., so rows are [n_prompt - 1, n_prompt - 1 + len(answer)).
        self._answer_slice = slice(n_prompt - 1, n_prompt - 1 + len(ans_ids))
        self._per_layer = []
        with torch.no_grad():
            self.model(torch.tensor([ids], device=self.device), output_attentions=True, use_cache=False)
        self._answer_slice = None

        if len(self._per_layer) != len(self.layers):
            raise RuntimeError(
                f"captured {len(self._per_layer)} layers, expected {len(self.layers)}; the "
                "attention module may not return weights in this transformers version"
            )
        out = np.zeros((len(self.layers), len(passages)), dtype=np.float32)
        for li, row in enumerate(self._per_layer):
            row = row.numpy()
            for pi, (a, b) in enumerate(tok_spans):
                out[li, pi] = row[a:b].sum()
        out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
        return out


def upper_half_mean(att: np.ndarray) -> np.ndarray:
    """Pre-registered H5 statistic: mean passage share over the upper half of layers."""
    return att[att.shape[0] // 2:].mean(axis=0)
