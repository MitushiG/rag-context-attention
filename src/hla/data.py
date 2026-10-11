"""Load NQ, HotpotQA and MuSiQue into one schema.

Every example is a dict:

    {
      "qid": str, "dataset": "nq" | "hotpotqa" | "musique",
      "question": str, "answers": [str, ...],
      "hops": int,                 # 1 for NQ, 2 for HotpotQA, 2-4 for MuSiQue
      "qtype": str | None,         # HotpotQA 'bridge'/'comparison'; MuSiQue id prefix
      "passages": [{"title", "text", "is_gold": bool, "hop": int | None}, ...],
    }

``passages`` is the question's own candidate set (gold + distractors). ``hop`` gives the
order in which a gold passage is needed (1 = first hop); it is None for distractors and
for gold passages whose order is not defined (HotpotQA comparison questions).
"""
from __future__ import annotations

import gzip
import json
import random
import re
import urllib.request
from pathlib import Path

RAW = Path("data/raw")

HOTPOT_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
# 20-document NQ-Open release from "Lost in the Middle" (Liu et al., 2023).
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


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


# --------------------------------------------------------------------------- NQ

def load_nq(raw_dir: Path = RAW) -> list[dict]:
    path = _download(NQ_LITM_URL, raw_dir / "nq-open-20docs.jsonl.gz")
    out = []
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            passages = [
                {
                    "title": c["title"],
                    "text": c["text"],
                    "is_gold": bool(c.get("isgold", False)),
                    "hop": 1 if c.get("isgold", False) else None,
                }
                for c in r["ctxs"]
            ]
            if sum(p["is_gold"] for p in passages) != 1:
                continue
            out.append(
                {
                    "qid": f"nq-{i}",
                    "dataset": "nq",
                    "question": r["question"],
                    "answers": list(r["answers"]),
                    "hops": 1,
                    "qtype": None,
                    "passages": passages,
                }
            )
    return out


# --------------------------------------------------------------------------- HotpotQA

def load_hotpotqa(raw_dir: Path = RAW) -> list[dict]:
    path = _download(HOTPOT_URL, raw_dir / "hotpot_dev_distractor_v1.json")
    data = json.loads(path.read_text())
    out = []
    for r in data:
        gold_titles = {t for t, _ in r["supporting_facts"]}
        passages = [
            {"title": t, "text": " ".join(s.strip() for s in sents), "is_gold": t in gold_titles, "hop": None}
            for t, sents in r["context"]
        ]
        golds = [p for p in passages if p["is_gold"]]
        if len(golds) != 2:
            continue
        if r["type"] == "bridge":
            # Hop 1 is the passage the question names directly. Use title overlap with the
            # question; ties leave hop order undefined (excluded from per-hop analyses).
            q = _tokens(r["question"])
            o = [len(_tokens(g["title"]) & q) / max(1, len(_tokens(g["title"]))) for g in golds]
            if o[0] != o[1]:
                first = 0 if o[0] > o[1] else 1
                golds[first]["hop"], golds[1 - first]["hop"] = 1, 2
        else:  # comparison: both entities are named in the question
            for g in golds:
                g["hop"] = 1
        out.append(
            {
                "qid": f"hotpotqa-{r['_id']}",
                "dataset": "hotpotqa",
                "question": r["question"],
                "answers": [r["answer"]],
                "hops": 2,
                "qtype": r["type"],
                "passages": passages,
            }
        )
    return out


# --------------------------------------------------------------------------- MuSiQue

def load_musique(raw_dir: Path = RAW) -> list[dict]:
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
        if not r.get("answerable", True):
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
        out.append(
            {
                "qid": f"musique-{r['id']}",
                "dataset": "musique",
                "question": r["question"],
                "answers": [r["answer"], *r.get("answer_aliases", [])],
                "hops": len(r["question_decomposition"]),
                "qtype": r["id"].split("__")[0],  # e.g. '2hop', '3hop1', '4hop3'
                "passages": passages,
            }
        )
    return out


# --------------------------------------------------------------------------- sampling

def sample(examples: list[dict], n: int, seed: int = 13) -> list[dict]:
    rng = random.Random(seed)
    return rng.sample(examples, min(n, len(examples)))


def sample_musique(examples: list[dict], per_hop: dict[int, int], seed: int = 13) -> list[dict]:
    """Stratified sample by hop count, e.g. {2: 200, 3: 150, 4: 150}."""
    rng = random.Random(seed)
    out = []
    for h, n in sorted(per_hop.items()):
        pool = [e for e in examples if e["hops"] == h]
        out.extend(rng.sample(pool, min(n, len(pool))))
    return out


def write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
