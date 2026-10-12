"""Shared helpers for the experiment scripts."""
from __future__ import annotations

from pathlib import Path

from .data import read_jsonl, write_jsonl
from .metrics import exact_match, f1_score

PROCESSED = Path("data/processed")
SAMPLES = Path("data/samples")
RESULTS = Path("results")


def load_examples(ids_file: str) -> list[dict]:
    """Examples listed in data/samples/<ids_file>, in file order."""
    ids = (SAMPLES / ids_file).read_text().split()
    by_id = {e["qid"]: e for e in read_jsonl(PROCESSED / "examples.jsonl")}
    return [by_id[q] for q in ids]


def load_corpus() -> list[dict]:
    return read_jsonl(PROCESSED / "corpus.jsonl")


def score(records: list[dict], examples: list[dict]) -> list[dict]:
    """Attach EM/F1 and the dataset name to pipeline outputs."""
    by_id = {e["qid"]: e for e in examples}
    return [
        {**r, "dataset": by_id[r["qid"]]["dataset"],
         "em": exact_match(r["pred"], by_id[r["qid"]]["answers"]),
         "f1": f1_score(r["pred"], by_id[r["qid"]]["answers"])}
        for r in records
    ]


def save(rows: list[dict], *parts: str) -> Path:
    path = RESULTS.joinpath(*parts)
    write_jsonl(rows, path)
    print(f"wrote {len(rows)} rows -> {path}")
    return path
