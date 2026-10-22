"""Prompts and QA pipelines: direct answer, single-shot chain of thought, and IRCoT.

Every record carries the prediction, the passages the model saw, and the cost (LLM calls
and tokens), so Exp 2 can compare accuracy and cost from the same outputs.
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
    """User message for a direct answer. The attention probe rebuilds exactly this string."""
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
    outs = gen.generate(prompts, max_tokens=64, stop=["\n"])
    return [
        {
            "qid": ex["qid"], "pred": _clean(o.text), "mean_logprob": o.mean_logprob,
            "context_ids": [p["doc_id"] for p in (ctx or [])],
            "prompt_tokens": gen.count_tokens(prompt), "gen_tokens": o.n_tokens, "llm_calls": 1,
        }
        for ex, ctx, o, prompt in zip(examples, contexts, outs, prompts)
    ]


# --------------------------------------------------------------------------- chain of thought

ANSWER_RE = re.compile(r"so the answer is\s*:?\s*(.+)", re.I)

COT_INSTRUCTION = (
    "Answer the question using the documents. Think step by step in a few short sentences, "
    "then finish with the sentence 'So the answer is: <short answer>.'"
)
NEXT_STEP_INSTRUCTION = (
    "Answer the question by reasoning step by step using the documents. Write only the NEXT "
    "single sentence of reasoning. When you know the final answer, write the sentence "
    "'So the answer is: <short answer>.'"
)


def _extract(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    return _clean(m.group(1)) if m else None


def _finish(gen: Generator, examples, records, contexts) -> None:
    """Questions whose reasoning never stated an answer get one direct-answer call over the
    same passages (recorded as ``fallback``), so every method returns an answer."""
    todo = [i for i, r in enumerate(records) if r["pred"] is None]
    if not todo:
        return
    outs = answer_with_contexts(gen, [examples[i] for i in todo], [contexts[i] for i in todo])
    for i, o in zip(todo, outs):
        r = records[i]
        r["pred"], r["fallback"] = o["pred"], True
        r["llm_calls"] += 1
        r["prompt_tokens"] += o["prompt_tokens"]
        r["gen_tokens"] += o["gen_tokens"]


def single_shot_cot(gen: Generator, examples: list[dict], contexts: list[list[dict]]) -> list[dict]:
    """Same passages as single-shot RAG, but the model reasons before answering."""
    prompts = [gen.chat_prompt(f"{COT_INSTRUCTION}\n\n{format_passages(c)}\n\nQuestion: {ex['question']}")
               for ex, c in zip(examples, contexts)]
    outs = gen.generate(prompts, max_tokens=256)
    records = [
        {"qid": ex["qid"], "pred": _extract(o.text), "reasoning": o.text, "fallback": False,
         "context_ids": [p["doc_id"] for p in c], "llm_calls": 1,
         "prompt_tokens": gen.count_tokens(pr), "gen_tokens": o.n_tokens}
        for ex, c, o, pr in zip(examples, contexts, outs, prompts)
    ]
    _finish(gen, examples, records, contexts)
    return records


def ircot(gen: Generator, retriever, docs: dict, examples: list[dict], k_init: int = 5,
          k_step: int = 3, max_steps: int = 5, max_passages: int = 15) -> list[dict]:
    """Interleaving Retrieval with Chain-of-Thought (Trivedi et al., 2023), simplified.

    Loop: write one reasoning sentence -> use that sentence as the next retrieval query
    -> add up to ``k_step`` new passages -> repeat, until the model states an answer or
    ``max_steps`` is reached. The reasoning sentence names the *bridge entity* (e.g. the
    director's name) that the question did not, so the hop-2 passage becomes retrievable.

    ``found_by_step[s]`` lists which gold passages were in context after step s (0 = the
    initial retrieval), which is how Exp 2 attributes gains to retrieval.
    """
    n = len(examples)
    init = retriever.search([ex["question"] for ex in examples], k=k_init)
    st = [{"ids": list(r), "steps": [], "pred": None, "calls": 0, "ptok": 0, "gtok": 0} for r in init]
    found = [[[g in s["ids"] for g in ex["gold_ids"]]] for ex, s in zip(examples, st)]
    for _ in range(max_steps):
        active = [i for i in range(n) if st[i]["pred"] is None]
        if not active:
            break
        prompts = []
        for i in active:
            so_far = " ".join(st[i]["steps"]) or "(none)"
            body = (f"{NEXT_STEP_INSTRUCTION}\n\n{format_passages([docs[d] for d in st[i]['ids']])}\n\n"
                    f"Question: {examples[i]['question']}\nReasoning so far: {so_far}\nNext sentence:")
            prompts.append(gen.chat_prompt(body))
        outs = gen.generate(prompts, max_tokens=80, stop=["\n"])
        queries = []
        for i, o, pr in zip(active, outs, prompts):
            s = st[i]
            s["calls"] += 1
            s["ptok"] += gen.count_tokens(pr)
            s["gtok"] += o.n_tokens
            sent = o.text.strip()
            s["steps"].append(sent)
            s["pred"] = _extract(sent)
            if s["pred"] is None:
                queries.append((i, sent or examples[i]["question"]))
        if queries:
            new = retriever.search([q for _, q in queries], k=k_step + max_passages)
            for (i, _), ranked in zip(queries, new):
                s = st[i]
                fresh = [d for d in ranked if d not in s["ids"]][:k_step]
                s["ids"] = (s["ids"] + fresh)[:max_passages]
        for i in active:
            found[i].append([g in st[i]["ids"] for g in examples[i]["gold_ids"]])

    records = [
        {"qid": ex["qid"], "pred": s["pred"], "reasoning": " ".join(s["steps"]), "fallback": False,
         "context_ids": s["ids"], "llm_calls": s["calls"], "prompt_tokens": s["ptok"],
         "gen_tokens": s["gtok"], "found_by_step": f}
        for ex, s, f in zip(examples, st, found)
    ]
    _finish(gen, examples, records, [[docs[d] for d in s["ids"]] for s in st])
    return records
