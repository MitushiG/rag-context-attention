"""Download, sample (seed 13) and pool the corpus.

Writes
  data/samples/nq_ids.txt          200 NQ questions (Exp 1 and Exp 2)
  data/samples/musique_ids.txt     200 MuSiQue 2-hop questions (Exp 2)
  data/samples/attention_ids.txt   the first 100 NQ questions (Exp 1 attention probe)
  data/processed/examples.jsonl, corpus.jsonl   (git-ignored, regenerated from raw)

Commit the id files before running anything on a GPU.
"""
from hla.corpus import build_corpus
from hla.data import load_musique_2hop, load_nq, sample, write_jsonl
from hla.runtime import PROCESSED, SAMPLES

SEED = 13
N = 200
N_ATTENTION = 100


def main():
    nq_all, mq_all = load_nq(), load_musique_2hop()
    # The corpus pools every question's passages, not only the sampled ones, so retrieval
    # and the long contexts draw on a realistic number of passages.
    corpus, examples = build_corpus(nq_all + mq_all)
    by_id = {e["qid"]: e for e in examples}
    nq = sample([e for e in examples if e["dataset"] == "nq"], N, SEED)
    mq = sample([e for e in examples if e["dataset"] == "musique"], N, SEED)

    write_jsonl([by_id[e["qid"]] for e in nq + mq], PROCESSED / "examples.jsonl")
    write_jsonl(corpus, PROCESSED / "corpus.jsonl")
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for name, ids in [("nq", [e["qid"] for e in nq]), ("musique", [e["qid"] for e in mq]),
                      ("attention", [e["qid"] for e in nq[:N_ATTENTION]])]:
        (SAMPLES / f"{name}_ids.txt").write_text("\n".join(ids) + "\n")
    print(f"NQ {len(nq_all)} -> {len(nq)}; MuSiQue 2-hop {len(mq_all)} -> {len(mq)}; "
          f"corpus passages {len(corpus)}")


if __name__ == "__main__":
    main()
