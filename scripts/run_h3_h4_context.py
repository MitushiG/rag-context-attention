"""H3 (long context vs RAG) and H4 (position, gap).

  python scripts/run_h3_h4_context.py --part h3
  python scripts/run_h3_h4_context.py --part h4a
  python scripts/run_h3_h4_context.py --part h4b
"""
import argparse

from hla.contexts import gap_context, length_context, position_context
from hla.generation import Generator
from hla.pipelines import answer_with_contexts
from hla.runtime import build_retriever, load_corpus, load_examples, save, score

# Nominal 32K leaves room for the instruction, question and answer inside Qwen2.5's
# 32,768-token window.
LENGTHS = {"2K": 2_000, "8K": 8_000, "16K": 16_000, "32K": 31_000}
POSITIONS = (1, 5, 10, 15, 20)
GAPS = (0, 5, 10, 18)


def run_h3(gen, corpus):
    examples = load_examples("h3_ids.txt")
    retriever = build_retriever("hybrid", corpus)
    rows = []
    for name, target in LENGTHS.items():
        ctxs = [length_context(e, target, gen.count_tokens, corpus) for e in examples]
        n_tok = [sum(gen.count_tokens(p["title"] + " " + p["text"]) + 8 for p in c) for c in ctxs]

        long = score(answer_with_contexts(gen, examples, ctxs), examples)
        allowed = [{p["doc_id"] for p in c} for c in ctxs]
        ranked = retriever.search([e["question"] for e in examples], k=5, allowed=allowed)
        docs = {p["doc_id"]: p for c in ctxs for p in c}
        rag = score(answer_with_contexts(gen, examples, [[docs[d] for d in r] for r in ranked]), examples)

        for cond, recs in (("long_context", long), ("rag", rag)):
            for r, t in zip(recs, n_tok):
                rows.append({**r, "condition": cond, "length_name": name, "target_tokens": target, "context_tokens": t})
        print(name, "long EM", sum(r["em"] for r in long) / len(long), "rag EM", sum(r["em"] for r in rag) / len(rag))
    save(rows, "h3", "length.jsonl")


def run_h4a(gen, corpus):
    examples = load_examples("nq_ids.txt")
    rows = []
    for pos in POSITIONS:
        ctxs = [position_context(e, pos, pool=corpus) for e in examples]
        recs = score(answer_with_contexts(gen, examples, ctxs), examples)
        rows += [{**r, "gold_position": pos} for r in recs]
        print("position", pos, "EM", sum(r["em"] for r in recs) / len(recs))
    save(rows, "h4", "position.jsonl")


def run_h4b(gen, corpus):
    examples = [e for e in load_examples("hotpotqa_ids.txt") if len(e["gold_ids"]) == 2]
    rows = []
    for d in GAPS:
        ctxs = [gap_context(e, d, pool=corpus) for e in examples]
        recs = score(answer_with_contexts(gen, examples, ctxs), examples)
        rows += [{**r, "gap": d} for r in recs]
        print("gap", d, "EM", sum(r["em"] for r in recs) / len(recs))
    save(rows, "h4", "gap.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["h3", "h4a", "h4b"], required=True)
    args = ap.parse_args()
    corpus = load_corpus()
    if args.part == "h3":
        gen = Generator(max_model_len=32768, gpu_memory_utilization=0.80)
        run_h3(gen, corpus)
    else:
        gen = Generator(max_model_len=8192)
        {"h4a": run_h4a, "h4b": run_h4b}[args.part](gen, corpus)


if __name__ == "__main__":
    main()
