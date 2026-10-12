"""Exp 2: attention share on the gold passages, for the answers from Exp 1 and Exp 3.

  python scripts/run_exp2_attention.py --split pilot --check --limit 5
  python scripts/run_exp2_attention.py --split main

Run after run_exp1_context.py and run_exp3_position.py (it needs the model's answers and
rebuilds exactly the contexts it saw). Uses transformers, not vLLM: free the GPU first.
Rows are appended as they finish, so a disconnected Colab session can resume.
"""
import argparse
import json

import numpy as np

from hla.attention import AttentionProbe, upper_half_mean
from hla.corpus import labelled
from hla.data import read_jsonl
from hla.runtime import RESULTS, SPLITS, load_corpus, load_examples

EXP1_CONDITIONS = ("rag", "4K", "12K", "28K")


def jobs(split: str):
    """(source, condition label, Exp 1/3 row) for every row with both golds in context."""
    e1 = read_jsonl(RESULTS / split / "exp1.jsonl")
    e3 = read_jsonl(RESULTS / split / "exp3.jsonl")
    out = [("exp1", r["condition"], r) for r in e1 if r["condition"] in EXP1_CONDITIONS and r["n_gold"] == 2]
    out += [("exp3", f"{r['location']}-{r['separation']}", r) for r in e3]
    return sorted(out, key=lambda j: j[2]["context_tokens"])  # short contexts first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--limit", type=int, default=None, help="questions per condition (debug)")
    ap.add_argument("--check", action="store_true", help="verify the probe against eager attention first")
    args = ap.parse_args()

    examples = {e["qid"]: e for e in load_examples(args.split)}
    docs = {d["doc_id"]: d for d in load_corpus()}
    probe = AttentionProbe.from_pretrained()

    todo = jobs(args.split)
    if args.check:
        for _, _, r in todo[:3]:
            ex = examples[r["qid"]]
            ids, rows, _ = probe.encode(ex["question"], [labelled(ex, docs[d]) for d in r["context_ids"]], r["pred"] or " ")
            print(f"probe vs eager: max abs diff {probe.check_against_eager(ids, rows):.2e}")

    path = RESULTS / args.split / "exp2.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["source"], r["condition"], r["qid"]) for r in read_jsonl(path)} if path.exists() else set()
    per_cond: dict[str, int] = {}
    with open(path, "a") as f:
        for source, cond, r in todo:
            if args.limit:
                per_cond[cond] = per_cond.get(cond, 0) + 1
                if per_cond[cond] > args.limit:
                    continue
            if (source, cond, r["qid"]) in done:
                continue
            ex = examples[r["qid"]]
            ctx = [labelled(ex, docs[d]) for d in r["context_ids"]]
            att = probe.passage_attention(ex["question"], ctx, r["pred"] or " ")  # [layers, passages]
            gold_idx = [i for i, p in enumerate(ctx) if p["is_gold"]]
            hop_idx = {p["hop"]: i for i, p in enumerate(ctx) if p["is_gold"]}
            share = upper_half_mean(att)
            n_tok = np.array([len(p["text"].split()) + len(p["title"].split()) for p in ctx], float)
            f.write(json.dumps({
                "source": source, "condition": cond, "qid": r["qid"], "em": r["em"],
                "context_tokens": r["context_tokens"], "n_passages": len(ctx),
                "location": r.get("location"), "separation": r.get("separation"),
                # pre-registered statistic: upper-half-layer share on both gold passages
                "gold_share": float(share[gold_idx].sum()),
                "hop1_share": float(share[hop_idx[1]]), "hop2_share": float(share[hop_idx[2]]),
                # share the golds would get if attention were proportional to length (words)
                "gold_length_share": float(n_tok[gold_idx].sum() / n_tok.sum()),
                "gold_share_per_layer": att[:, gold_idx].sum(axis=1).round(5).tolist(),
            }) + "\n")
            f.flush()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
