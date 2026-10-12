"""The one QA prompt every condition uses, and batched answering with a fixed context.

Exp 1 and Exp 3 differ only in *which passages* are passed and in what order; the
instruction, formatting, decoding and scoring are identical.
"""
from __future__ import annotations

import re

from .generation import Generator

QA_INSTRUCTION = (
    "Answer the question. Respond with only the short answer (a few words, e.g. a name, "
    "date or number). Do not explain."
)


def format_passage(i: int, p: dict) -> str:
    return f"Document [{i}] (Title: {p['title']}) {p['text']}"


def format_passages(passages: list[dict]) -> str:
    return "\n\n".join(format_passage(i, p) for i, p in enumerate(passages, start=1))


def qa_body(question: str, passages: list[dict] | None) -> str:
    """User message. The attention probe rebuilds exactly this string."""
    if passages:
        return f"{QA_INSTRUCTION} Use the documents below.\n\n{format_passages(passages)}\n\nQuestion: {question}\nAnswer:"
    return f"{QA_INSTRUCTION}\n\nQuestion: {question}\nAnswer:"


def _clean(ans: str) -> str:
    ans = ans.strip().split("\n")[0]
    ans = re.sub(r"^(answer\s*:\s*)", "", ans, flags=re.I)
    return ans.strip().rstrip(".")


def answer_with_contexts(gen: Generator, examples: list[dict], contexts: list[list[dict] | None]) -> list[dict]:
    """One LLM call per question with a fixed context (None = closed-book)."""
    prompts = [gen.chat_prompt(qa_body(ex["question"], ctx)) for ex, ctx in zip(examples, contexts)]
    outs = gen.generate(prompts, max_tokens=32, stop=["\n"])
    return [
        {
            "qid": ex["qid"], "pred": _clean(o.text), "mean_logprob": o.mean_logprob,
            "context_ids": [p["doc_id"] for p in (ctx or [])],
            "prompt_tokens": gen.count_tokens(prompt),
        }
        for ex, ctx, o, prompt in zip(examples, contexts, outs, prompts)
    ]
