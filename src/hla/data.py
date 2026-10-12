"""Load MuSiQue-Ans 2-hop questions into one schema.

Every example is a dict:

    {
      "qid": str, "question": str, "answers": [str, ...],
      "passages": [{"title", "text", "is_gold": bool, "hop": int | None}, ...],
    }

MuSiQue was built so that its multi-hop questions cannot be answered from one passage
(Trivedi et al., 2022), which is what the position and attention experiments need. Each
2-hop question comes with exactly 2 gold passages and 18 hard distractors. ``hop`` gives
the order in which a gold passage is needed (1 = first hop); None for distractors.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

RAW = Path("data/raw")
MUSIQUE_FILE = "musique_ans_v1.0_dev.jsonl"  # see README: manual download


def load_musique_2hop(raw_dir: Path = RAW) -> list[dict]:
    path = raw_dir / MUSIQUE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download MuSiQue v1.0 from "
            "https://github.com/StonyBrookNLP/musique (download_data.sh) and copy "
            f"{MUSIQUE_FILE} into {raw_dir}/."
        )
    out = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if not r["id"].startswith("2hop") or not r.get("answerable", True):
            continue
        hop_of_para = {}
        for h, step in enumerate(r["question_decomposition"], start=1):
            idx = step.get("paragraph_support_idx")
            if idx is not None:
                hop_of_para.setdefault(idx, h)
        passages = [
            {
                "title": p["title"],
                "text": p["paragraph_text"],
                "is_gold": bool(p["is_supporting"]),
                "hop": hop_of_para.get(p["idx"]) if p["is_supporting"] else None,
            }
            for p in r["paragraphs"]
        ]
        golds = [p for p in passages if p["is_gold"]]
        # Experiment 3 needs two gold passages with a defined hop order and 18 distractors.
        if sorted(g["hop"] or 0 for g in golds) != [1, 2] or len(passages) - len(golds) < 18:
            continue
        out.append(
            {
                "qid": f"musique-{r['id']}",
                "question": r["question"],
                "answers": [r["answer"], *r.get("answer_aliases", [])],
                "passages": passages,
            }
        )
    return out


def split_pilot(examples: list[dict], n_pilot: int, seed: int = 13) -> tuple[list[dict], list[dict]]:
    """Shuffle once (seeded). The first ``n_pilot`` are the pilot; the rest, in order,
    are the main pool. The main sample is a prefix of the pool whose length is set by
    the power analysis, so the sample does not depend on what the pilot showed."""
    order = list(examples)
    random.Random(seed).shuffle(order)
    return order[:n_pilot], order[n_pilot:]


def write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
