"""QA pipelines: closed-book, reading a given context, single-shot RAG, IRCoT.

Every pipeline returns one record per question with the prediction, the passages the
model saw, the answer confidence and the cost (LLM calls), so H2 and H7 can be scored
from the same outputs.
"""
from __future__ import annotations

import re

from .generation import Generator

QA_INSTRUCTION = (
    "Answer the question. Respond with only the short answer (a few words, e.g. a name, "
    "date or number). Do not explain."
)


def format_passages(passages: list[dict]) -> str:
    return "\n\n".join(
        f"Document [{i}] (Title: {p['title']}) {p['text']}" for i, p in enumerate(passages, start=1)
    )


def qa_prompt(gen: Generator, question: str, passages: list[dict] | None) -> str:
    if passages:
        body = f"{QA_INSTRUCTION} Use the documents below.\n\n{format_passages(passages)}\n\nQuestion: {question}\nAnswer:"
    else:
        body = f"{QA_INSTRUCTION}\n\nQuestion: {question}\nAnswer:"
    return gen.chat_prompt(body)


def _clean(ans: str) -> str:
    ans = ans.strip().split("\n")[0]
    ans = re.sub(r"^(answer\s*:\s*)", "", ans, flags=re.I)
    return ans.strip().rstrip(".")


def answer_with_contexts(gen: Generator, examples: list[dict], contexts: list[list[dict] | None]) -> list[dict]:
    """One LLM call per question with a fixed context (None = closed-book)."""
    prompts = [qa_prompt(gen, ex["question"], ctx) for ex, ctx in zip(examples, contexts)]
    outs = gen.generate(prompts, max_tokens=32, stop=["\n"])
    return [
        {
            "qid": ex["qid"], "pred": _clean(o.text), "mean_logprob": o.mean_logprob,
            "confidence": o.confidence, "llm_calls": 1,
            "context_ids": [p.get("doc_id") for p in (ctx or [])],
        }
        for ex, ctx, o in zip(examples, contexts, outs)
    ]


def single_shot_rag(gen: Generator, retriever, docs: dict, examples: list[dict], k: int = 5) -> list[dict]:
    ranked = retriever.search([ex["question"] for ex in examples], k=k)
    contexts = [[docs[d] for d in r] for r in ranked]
    return answer_with_contexts(gen, examples, contexts)


# --------------------------------------------------------------------------- IRCoT

COT_INSTRUCTION = (
    "Answer the question by reasoning step by step using the documents. Write only the NEXT "
    "single sentence of reasoning. When you know the final answer, write the sentence "
    "'So the answer is: <short answer>.'"
)
ANSWER_RE = re.compile(r"so the answer is\s*:?\s*(.+)", re.I)


def _cot_prompt(gen: Generator, question: str, passages: list[dict], steps: list[str]) -> str:
    so_far = " ".join(steps)
    body = (
        f"{COT_INSTRUCTION}\n\n{format_passages(passages)}\n\nQuestion: {question}\n"
        f"Reasoning so far: {so_far if so_far else '(none)'}\nNext sentence:"
    )
    return gen.chat_prompt(body)


def ircot(gen: Generator, retriever, docs: dict, examples: list[dict], k_init: int = 5,
          k_step: int = 3, max_steps: int = 4, max_passages: int = 15) -> list[dict]:
    """Interleaving Retrieval with Chain-of-Thought (Trivedi et al., 2023), simplified.

    Loop: write one reasoning sentence -> use that sentence as the next retrieval query
    -> add new passages -> repeat, until the model states an answer or max_steps.
    The key idea: the reasoning sentence contains the *bridge entity* (e.g. the
    director's name) that the original question did not, so hop-2 evidence becomes
    retrievable.
    """
    n = len(examples)
    init = retriever.search([ex["question"] for ex in examples], k=k_init)
    state = [
        {"ids": list(r), "steps": [], "answer": None, "calls": 0} for r in init
    ]
    for _ in range(max_steps):
        active = [i for i in range(n) if state[i]["answer"] is None]
        if not active:
            break
        prompts = [
            _cot_prompt(gen, examples[i]["question"], [docs[d] for d in state[i]["ids"]], state[i]["steps"])
            for i in active
        ]
        outs = gen.generate(prompts, max_tokens=80, stop=["\n"])
        queries = []
        for i, o in zip(active, outs):
            s = state[i]
            s["calls"] += 1
            sent = o.text.strip()
            s["steps"].append(sent)
            m = ANSWER_RE.search(sent)
            if m:
                s["answer"] = _clean(m.group(1))
            queries.append(sent or examples[i]["question"])
        still = [(i, q) for i, q in zip(active, queries) if state[i]["answer"] is None]
        if still:
            new = retriever.search([q for _, q in still], k=k_step + max_passages)
            for (i, _), ranked in zip(still, new):
                s = state[i]
                fresh = [d for d in ranked if d not in s["ids"]][:k_step]
                s["ids"] = (s["ids"] + fresh)[:max_passages]

    # Final short answer with the collected evidence. Always run, so every method's
    # answer comes from the same QA prompt and confidence is comparable across methods.
    finals = answer_with_contexts(gen, examples, [[docs[d] for d in s["ids"]] for s in state])
    out = []
    for ex, s, f in zip(examples, state, finals):
        out.append({
            **f,
            "cot_answer": s["answer"],
            "steps": s["steps"],
            "llm_calls": s["calls"] + 1,
            "context_ids": s["ids"],
        })
    return out
