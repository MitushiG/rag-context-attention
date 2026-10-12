"""Exp 3: does separating the two gold passages hurt, at a fixed location and length?

  python scripts/run_exp3_position.py --split pilot
  python scripts/run_exp3_position.py --split main

3 locations (early / middle / late) x 2 separations (close / far); 20 passages; both
gold passages are always present. See hla.contexts.pair_context for the slot layout.
"""
import argparse
import itertools

from hla.contexts import LOCATIONS, SEPARATIONS, pair_context, pair_slots
from hla.generation import Generator
from hla.pipelines import answer_with_contexts, format_passages
from hla.runtime import SPLITS, load_examples, save, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N questions (debug)")
    args = ap.parse_args()

    examples = load_examples(args.split)[: args.limit]
    gen = Generator(max_model_len=8192)
    rows = []
    for loc, sep in itertools.product(LOCATIONS, SEPARATIONS):
        ctxs = [pair_context(e, loc, sep) for e in examples]
        recs = score(answer_with_contexts(gen, examples, ctxs), examples)
        a, b = pair_slots(loc, sep)
        for ctx, r in zip(ctxs, recs):
            assert ctx[a - 1]["is_gold"] and ctx[b - 1]["is_gold"]
            rows.append({
                **r, "location": loc, "separation": sep, "far": int(sep == "far"),
                "gold_slots": [a, b], "hop1_first": ctx[a - 1]["hop"] == 1,
                "context_tokens": gen.count_tokens(format_passages(ctx)),
            })
        print(f"{loc:6s} {sep:5s} EM {sum(r['em'] for r in recs) / len(recs):.3f}")
    save(rows, args.split, "exp3.jsonl")


if __name__ == "__main__":
    main()
