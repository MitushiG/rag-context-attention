"""Exp 1 attention: share of the answer tokens' attention that lands on the gold passage.

  python scripts/run_exp1_attention.py --check --limit 2   # debug
  python scripts/run_exp1_attention.py

Run after run_exp1_length.py (it needs the model's answers and the saved context ids).
Covers the 100 questions in attention_ids.txt, every condition where the gold passage is
in context except RAG. Uses transformers, not vLLM: free the GPU first. Rows are appended
as they finish, so a disconnected Colab session can resume.
"""
import argparse
import json

import numpy as np

from hla.attention import AttentionProbe, upper_half_mean
from hla.corpus import labelled
from hla.data import read_jsonl
from hla.runtime import RESULTS, SAMPLES, load_corpus, load_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="questions per condition (debug)")
    ap.add_argument("--check", action="store_true", help="verify the probe against eager attention first")
    args = ap.parse_args()

    keep = set((SAMPLES / "attention_ids.txt").read_text().split())
    examples = {e["qid"]: e for e in load_examples("nq_ids.txt")}
    docs = {d["doc_id"]: d for d in load_corpus()}
    todo = [r for r in read_jsonl(RESULTS / "exp1" / "exp1.jsonl")
            if r["qid"] in keep and r["gold_in_context"] and not r["condition"].startswith("rag")]
    todo.sort(key=lambda r: r["context_tokens"])  # short contexts first
    probe = AttentionProbe.from_pretrained()

    def context(r):
        ex = examples[r["qid"]]
        return ex, [labelled(ex, docs[d]) for d in r["context_ids"]]

    if args.check:
        for r in [r for r in todo if r["n_passages"] > 1][:3]:
            ex, ctx = context(r)
            ids, rows, _ = probe.encode(ex["question"], ctx, r["pred"] or " ")
            print(f"probe vs eager: max abs diff {probe.check_against_eager(ids, rows):.2e}")

    path = RESULTS / "exp1" / "attention.jsonl"
    done = {(r["condition"], r["qid"]) for r in read_jsonl(path)} if path.exists() else set()
    per_cond: dict[str, int] = {}
    with open(path, "a") as f:
        for r in todo:
            per_cond[r["condition"]] = per_cond.get(r["condition"], 0) + 1
            if (args.limit and per_cond[r["condition"]] > args.limit) or (r["condition"], r["qid"]) in done:
                continue
            ex, ctx = context(r)
            att = probe.passage_attention(ex["question"], ctx, r["pred"] or " ")  # [layers, passages]
            g = r["gold_index"] - 1
            words = np.array([len((p["title"] + " " + p["text"]).split()) for p in ctx], float)
            f.write(json.dumps({
                "qid": r["qid"], "condition": r["condition"], "kind": r.get("kind"), "length": r.get("length"),
                "position": r.get("position"), "context_tokens": r["context_tokens"],
                "n_passages": len(ctx), "em": r["em"],
                # pre-registered statistic: upper-half-layer attention share on the gold passage
                "gold_share": float(upper_half_mean(att)[g]),
                # what the gold passage would get if attention were proportional to length
                "gold_length_share": float(words[g] / words.sum()),
                "gold_share_per_layer": att[:, g].round(5).tolist(),
            }) + "\n")
            f.flush()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
