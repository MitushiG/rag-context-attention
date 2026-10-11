"""FastAPI service for the confidence router (H7).

POST /answer {"question": "..."} ->
  answer from single-shot RAG; if its confidence is below the threshold chosen in
  scripts/run_h7_router.py, escalate to IRCoT.

The corpus is the closed benchmark corpus, so this is a demo of the serving path, not a
general QA service.

  uvicorn app.server:app --host 0.0.0.0 --port 8000
"""
import json
import time

from fastapi import FastAPI
from pydantic import BaseModel

from hla.generation import Generator
from hla.pipelines import ircot, single_shot_rag
from hla.runtime import RESULTS, build_retriever, load_corpus

app = FastAPI(title="Adaptive multi-hop RAG")
_state: dict = {}


class Query(BaseModel):
    question: str


@app.on_event("startup")
def _load():
    corpus = load_corpus()
    _state["docs"] = {d["doc_id"]: d for d in corpus}
    _state["retriever"] = build_retriever("hybrid", corpus)
    _state["gen"] = Generator(max_model_len=8192, gpu_memory_utilization=0.75)
    th = RESULTS / "h7" / "threshold.json"
    _state["threshold"] = json.loads(th.read_text())["threshold"] if th.exists() else 0.5


@app.get("/health")
def health():
    return {"ok": "gen" in _state, "threshold": _state.get("threshold")}


@app.post("/answer")
def answer(q: Query):
    t0 = time.perf_counter()
    ex = [{"qid": "live", "question": q.question}]
    gen, ret, docs = _state["gen"], _state["retriever"], _state["docs"]
    first = single_shot_rag(gen, ret, docs, ex, k=5)[0]
    route, final = "single_hop", first
    if first["confidence"] < _state["threshold"]:
        route, final = "multi_hop", ircot(gen, ret, docs, ex)[0]
    return {
        "answer": final["pred"],
        "route": route,
        "single_hop_confidence": first["confidence"],
        "llm_calls": first["llm_calls"] + (final["llm_calls"] if route == "multi_hop" else 0),
        "evidence": [docs[d]["title"] for d in final["context_ids"]],
        "reasoning": final.get("steps", []),
        "latency_s": round(time.perf_counter() - t0, 3),
    }
