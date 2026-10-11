"""H6: page-image retrieval (ColQwen2.5) vs OCR text retrieval on ViDoRe.

Foundations covered here
------------------------
* A vision-language model turns a page image into a sequence of patch embeddings (the
  vision encoder splits the image into patches; a projector maps them into the LLM's
  embedding space; the LLM contextualizes them).
* ColPali/ColQwen keep one vector per patch (multi-vector) instead of pooling to one
  vector, and score with late interaction (MaxSim): for each query token, take its best
  matching patch, then sum over query tokens. ``maxsim`` below implements this from
  scratch; ``check_against_engine`` compares it with colpali-engine's scorer.
"""
from __future__ import annotations

import numpy as np

# Grouping fixed a priori in HYPOTHESES.md. Verify names on the Hugging Face hub
# (https://huggingface.co/vidore) before running; drop any that no longer exist and log
# that as an amendment.
VIDORE_SUBSETS = {
    "vidore/infovqa_test_subsampled": "visual",
    "vidore/arxivqa_test_subsampled": "visual",
    "vidore/tatdqa_test": "visual",
    "vidore/tabfquad_test_subsampled": "visual",
    "vidore/shiftproject_test": "text",
    "vidore/syntheticDocQA_artificial_intelligence_test": "text",
    "vidore/syntheticDocQA_energy_test": "text",
    "vidore/syntheticDocQA_government_reports_test": "text",
    "vidore/syntheticDocQA_healthcare_industry_test": "text",
}


def load_vidore(name: str, split: str = "test"):
    """Return (pages, queries).

    pages: list of {"page_id", "image"}; queries: list of {"query", "page_id"}.
    ViDoRe v1 rows are (query, image, image_filename, ...); rows with an empty query
    are distractor pages that are still part of the corpus.
    """
    from datasets import load_dataset

    ds = load_dataset(name, split=split)
    pages, seen, queries = [], set(), []
    for r in ds:
        pid = r["image_filename"]
        if pid not in seen:
            seen.add(pid)
            pages.append({"page_id": pid, "image": r["image"]})
        if r.get("query"):
            queries.append({"query": r["query"], "page_id": pid})
    return pages, queries


def ocr_pages(pages: list[dict]) -> list[str]:
    """Tesseract OCR (needs `apt-get install tesseract-ocr` on Colab)."""
    import pytesseract

    return [pytesseract.image_to_string(p["image"].convert("RGB")) for p in pages]


# --------------------------------------------------------------------------- late interaction

def maxsim(q_embs: list, d_embs: list, device: str | None = None, q_chunk: int = 8):
    """Late-interaction scores, from scratch.

    q_embs: list of [n_q_tokens_i, dim] tensors; d_embs: list of [n_patches_j, dim].
    Returns a [n_queries, n_docs] tensor with
        score(q, d) = sum_over_q_tokens max_over_d_patches <q_t, d_p>.
    Padding is masked so ragged lengths are handled exactly. Queries are processed in
    chunks: the full similarity tensor is [queries, docs, q_tokens, patches], which for
    500 x 500 x 30 x 750 floats would be ~22 GB.
    """
    import torch

    if len(q_embs) > q_chunk:
        return torch.cat(
            [maxsim(q_embs[i:i + q_chunk], d_embs, device, q_chunk) for i in range(0, len(q_embs), q_chunk)]
        )
    return _maxsim_block(q_embs, d_embs, device)


def _maxsim_block(q_embs: list, d_embs: list, device: str | None = None):
    import torch

    device = device or (q_embs[0].device if len(q_embs) else "cpu")

    def pad(xs):
        n = max(x.shape[0] for x in xs)
        out = torch.zeros(len(xs), n, xs[0].shape[1], dtype=torch.float32, device=device)
        mask = torch.zeros(len(xs), n, dtype=torch.bool, device=device)
        for i, x in enumerate(xs):
            out[i, : x.shape[0]] = x.to(device, torch.float32)
            mask[i, : x.shape[0]] = True
        return out, mask

    Q, qm = pad(q_embs)
    D, dm = pad(d_embs)
    # sim[q, d, i, j] = <Q[q, i], D[d, j]>
    sim = torch.einsum("qid,pjd->qpij", Q, D)
    sim = sim.masked_fill(~dm[None, :, None, :], float("-inf"))
    best = sim.max(dim=-1).values                       # [nq, nd, q_tokens]
    best = best.masked_fill(~qm[:, None, :], 0.0)
    return best.sum(dim=-1)


class ColQwenRetriever:
    def __init__(self, model_name: str = "vidore/colqwen2.5-v0.2", device: str = "cuda"):
        import torch
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        self.torch = torch
        self.model = ColQwen2_5.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(model_name)
        self.device = device

    def embed_pages(self, images, batch_size: int = 4) -> list:
        out = []
        for i in range(0, len(images), batch_size):
            batch = self.processor.process_images(images[i:i + batch_size]).to(self.device)
            with self.torch.no_grad():
                emb = self.model(**batch)
            mask = batch["attention_mask"].bool()
            out.extend(e[m].cpu() for e, m in zip(emb, mask))
        return out

    def embed_queries(self, queries: list[str], batch_size: int = 32) -> list:
        out = []
        for i in range(0, len(queries), batch_size):
            batch = self.processor.process_queries(queries[i:i + batch_size]).to(self.device)
            with self.torch.no_grad():
                emb = self.model(**batch)
            mask = batch["attention_mask"].bool()
            out.extend(e[m].cpu() for e, m in zip(emb, mask))
        return out

    def check_against_engine(self, q_embs, d_embs, atol: float = 1e-2) -> bool:
        """Sanity check: our MaxSim matches colpali-engine's scorer."""
        ours = maxsim(q_embs, d_embs, device="cpu")
        ref = self.processor.score_multi_vector(q_embs, d_embs).float()
        return bool(np.allclose(ours.numpy(), ref.numpy(), atol=atol, rtol=1e-3))


def rank_pages(scores: np.ndarray, page_ids: list[str], k: int = 10) -> list[list[str]]:
    order = np.argsort(-scores, axis=1)[:, :k]
    return [[page_ids[j] for j in row] for row in order]
