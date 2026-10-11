"""H6: OCR-text retrieval vs ColQwen2.5 page-image retrieval on ViDoRe subsets."""
import argparse
import json

from hla.metrics import ndcg_at_k, recall_at_k
from hla.multimodal import VIDORE_SUBSETS, ColQwenRetriever, load_vidore, maxsim, ocr_pages
from hla.retrieval import BM25Retriever, DenseRetriever
from hla.runtime import RESULTS, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=list(VIDORE_SUBSETS))
    args = ap.parse_args()

    colqwen = ColQwenRetriever()
    rows = []
    for name in args.subsets:
        group = VIDORE_SUBSETS[name]
        pages, queries = load_vidore(name)
        page_ids = [p["page_id"] for p in pages]
        qtexts = [q["query"] for q in queries]
        print(f"{name}: {len(pages)} pages, {len(queries)} queries")

        # OCR (cached: it is the slow CPU step)
        cache = RESULTS / "h6" / "ocr" / (name.replace("/", "__") + ".json")
        if cache.exists():
            texts = json.loads(cache.read_text())
        else:
            texts = ocr_pages(pages)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(texts))
        corpus = [{"doc_id": pid, "title": "", "text": t} for pid, t in zip(page_ids, texts)]

        runs = {
            "ocr_bm25": BM25Retriever(corpus).search(qtexts, k=10),
            "ocr_dense": DenseRetriever(corpus).search(qtexts, k=10),
        }
        d_emb = colqwen.embed_pages([p["image"] for p in pages])
        q_emb = colqwen.embed_queries(qtexts)
        if not colqwen.check_against_engine(q_emb[:4], d_emb[:16]):
            print("WARNING: from-scratch MaxSim disagrees with colpali-engine on a sample")
        scores = maxsim(q_emb, d_emb, device="cuda").cpu().numpy()
        runs["colqwen"] = [[page_ids[j] for j in row] for row in (-scores).argsort(axis=1)[:, :10]]

        for method, ranked in runs.items():
            for q, r in zip(queries, ranked):
                rows.append({
                    "subset": name, "group": group, "query": q["query"], "method": method,
                    "ndcg@5": ndcg_at_k(r, {q["page_id"]: 1.0}, 5),
                    "recall@5": recall_at_k(r, [q["page_id"]], 5),
                })
    save(rows, "h6", "vidore.jsonl")


if __name__ == "__main__":
    main()
