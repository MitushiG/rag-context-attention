"""Shared helpers for the experiment scripts.

Results for the pilot and the main study go to separate folders, results/pilot/ and
results/main/, so the pilot can never leak into the confirmatory analysis.
"""
from __future__ import annotations

from pathlib import Path

from .data import read_jsonl, write_jsonl
from .metrics import exact_match, f1_score

PROCESSED = Path("data/processed")
SAMPLES = Path("data/samples")
RESULTS = Path("results")
SPLITS = ("pilot", "main")


def load_examples(split: str) -> list[dict]:
    """Examples of ``split`` in the order of data/samples/{split}_ids.txt."""
    ids = (SAMPLES / f"{split}_ids.txt").read_text().split()
    by_id = {e["qid"]: e for e in read_jsonl(PROCESSED / "examples.jsonl")}
    return [by_id[q] for q in ids]


def load_corpus() -> list[dict]:
    return read_jsonl(PROCESSED / "corpus.jsonl")


def score(records: list[dict], examples: list[dict]) -> list[dict]:
    """Attach EM/F1 to pipeline outputs."""
    by_id = {e["qid"]: e for e in examples}
    return [
        {**r, "em": exact_match(r["pred"], by_id[r["qid"]]["answers"]),
         "f1": f1_score(r["pred"], by_id[r["qid"]]["answers"])}
        for r in records
    ]


def save(rows: list[dict], split: str, name: str) -> Path:
    path = RESULTS / split / name
    write_jsonl(rows, path)
    print(f"wrote {len(rows)} rows -> {path}")
    return path
