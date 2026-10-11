"""Download, sample (seed 13) and pool the corpus.

Writes
  data/samples/{nq,hotpotqa,musique,h3}_ids.txt   (committed: the pre-registered sample)
  data/processed/examples.jsonl, corpus.jsonl     (git-ignored, regenerated from raw)
"""
import random

from hla.corpus import build_corpus
from hla.data import load_hotpotqa, load_musique, load_nq, sample, sample_musique, write_jsonl
from hla.runtime import PROCESSED, SAMPLES

SEED = 13


def main():
    nq = sample(load_nq(), 500, SEED)
    hotpot = sample(load_hotpotqa(), 500, SEED)
    musique = sample_musique(load_musique(), {2: 200, 3: 150, 4: 150}, SEED)
    examples = nq + hotpot + musique

    corpus, examples = build_corpus(examples)
    write_jsonl(examples, PROCESSED / "examples.jsonl")
    write_jsonl(corpus, PROCESSED / "corpus.jsonl")

    SAMPLES.mkdir(parents=True, exist_ok=True)
    for name, exs in [("nq", nq), ("hotpotqa", hotpot), ("musique", musique)]:
        (SAMPLES / f"{name}_ids.txt").write_text("\n".join(e["qid"] for e in exs) + "\n")

    # H3 subset: 100 per dataset
    rng = random.Random(SEED)
    h3 = [e["qid"] for exs in (nq, hotpot, musique) for e in rng.sample(exs, 100)]
    (SAMPLES / "h3_ids.txt").write_text("\n".join(h3) + "\n")

    print(f"examples: {len(examples)}  corpus passages: {len(corpus)}")
    for name, exs in [("nq", nq), ("hotpotqa", hotpot), ("musique", musique)]:
        print(f"  {name}: {len(exs)}")


if __name__ == "__main__":
    main()
