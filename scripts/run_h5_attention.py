"""H5: attention share on the gold passage, using the H4(a) contexts and answers.

Run after `run_h3_h4_context.py --part h4a` (needs the model's answers). Uses
transformers with eager attention, not vLLM, so free the GPU from other jobs first.
"""
import argparse

import numpy as np

from hla.attention import AttentionProbe, upper_half_mean
from hla.contexts import position_context
from hla.data import read_jsonl
from hla.runtime import RESULTS, load_corpus, load_examples, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="questions per position (debug)")
    args = ap.parse_args()

    examples = {e["qid"]: e for e in load_examples("nq_ids.txt")}
    corpus = load_corpus()
    h4 = read_jsonl(RESULTS / "h4" / "position.jsonl")
    probe = AttentionProbe()

    rows, per_layer = [], []
    by_pos: dict[int, int] = {}
    for r in h4:
        pos = r["gold_position"]
        if args.limit and by_pos.get(pos, 0) >= args.limit:
            continue
        by_pos[pos] = by_pos.get(pos, 0) + 1
        ex = examples[r["qid"]]
        ctx = position_context(ex, pos, pool=corpus)  # identical to H4(a) (seeded)
        att = probe.passage_attention(ex["question"], ctx, r["pred"] or " ")
        share = upper_half_mean(att)
        rows.append({
            "qid": r["qid"], "gold_position": pos, "em": r["em"],
            "gold_share": float(share[pos - 1]),
            "uniform_share": 1.0 / len(ctx),
        })
        per_layer.append(att[:, pos - 1])
    save(rows, "h5", "attention.jsonl")
    np.save(RESULTS / "h5" / "gold_share_per_layer.npy", np.stack(per_layer))


if __name__ == "__main__":
    main()
