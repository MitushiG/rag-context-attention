"""Load MuSiQue 2-hop dev, pool the corpus, and fix the question order (seed 13).

Writes
  data/samples/pilot_ids.txt       50 pilot questions (never used in the main analysis)
  data/samples/main_pool_ids.txt   every other question, in a fixed random order;
                                   scripts/power.py takes the main sample as a prefix
  data/processed/examples.jsonl, corpus.jsonl   (git-ignored, regenerated from raw)

Commit the two id files before running anything on a GPU.
"""
from hla.corpus import build_corpus
from hla.data import load_musique_2hop, split_pilot, write_jsonl
from hla.runtime import PROCESSED, SAMPLES

SEED = 13
N_PILOT = 50


def main():
    examples = load_musique_2hop()
    # The corpus pools every 2-hop dev question's passages, not only the sampled ones,
    # so the retriever and the expanded contexts see a realistic number of passages.
    corpus, examples = build_corpus(examples)
    write_jsonl(examples, PROCESSED / "examples.jsonl")
    write_jsonl(corpus, PROCESSED / "corpus.jsonl")

    pilot, pool = split_pilot(examples, N_PILOT, SEED)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    (SAMPLES / "pilot_ids.txt").write_text("\n".join(e["qid"] for e in pilot) + "\n")
    (SAMPLES / "main_pool_ids.txt").write_text("\n".join(e["qid"] for e in pool) + "\n")
    print(f"2-hop questions: {len(examples)}  pilot: {len(pilot)}  main pool: {len(pool)}  "
          f"corpus passages: {len(corpus)}")


if __name__ == "__main__":
    main()
