"""Load NQ-Open (single-hop) and MuSiQue-Ans 2-hop questions into one schema.

Every example is a dict:

    {
      "qid": str, "dataset": "nq" | "musique", "question": str, "answers": [str, ...],
      "passages": [{"title", "text", "is_gold": bool, "hop": int | None}, ...],
    }

``passages`` is the question's own candidate set (gold + distractors). ``hop`` gives the
order in which a gold passage is needed (1 = first hop); None for distractors.

- NQ-Open, "Lost in the Middle" release (Liu et al., 2023): 20 passages per question,
  exactly one of which is gold; the other 19 were retrieved for the question and do not
  contain the answer.
- MuSiQue 2-hop (Trivedi et al., 2022): built so one passage is not enough; 2 gold
  passages (hop 1, hop 2) and 18 distractors per question.
"""
from __future__ import annotations

import gzip
import json
import random
import urllib.request
from pathlib import Path

RAW = Path("data/raw")
NQ_LITM_URL = (
    "https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/main/"
    "qa_data/20_total_documents/nq-open-20_total_documents_gold_at_0.jsonl.gz"
)
MUSIQUE_FILE = "musique_ans_v1.0_dev.jsonl"  # see README: manual download


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_nq(raw_dir: Path = RAW) -> list[dict]:
    path = _download(NQ_LITM_URL, raw_dir / "nq-open-20docs.jsonl.gz")
    out = []
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            passages = [
                {"title": c["title"], "text": c["text"], "is_gold": bool(c.get("isgold", False)),
                 "hop": 1 if c.get("isgold", False) else None}
                for c in r["ctxs"]
            ]
            if sum(p["is_gold"] for p in passages) != 1:
                continue
            out.append({"qid": f"nq-{i}", "dataset": "nq", "question": r["question"],
                        "answers": list(r["answers"]), "passages": passages})
    return out


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
        # Keep questions with one hop-1 and one hop-2 gold passage (per-hop recall in Exp 2).
        if sorted(g["hop"] or 0 for g in golds) != [1, 2]:
            continue
        out.append(
            {
                "qid": f"musique-{r['id']}",
                "dataset": "musique",
                "question": r["question"],
                "answers": [r["answer"], *r.get("answer_aliases", [])],
                "passages": passages,
            }
        )
    return out


def sample(examples: list[dict], n: int, seed: int = 13) -> list[dict]:
    return random.Random(seed).sample(examples, min(n, len(examples)))


def write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
