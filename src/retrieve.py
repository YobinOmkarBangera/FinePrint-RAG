import json
import re
import sys
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_DIR / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = PROJECT_DIR / "data" / "index"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
# bge models expect this prefix on questions (not on the chunks)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(text):
    """Lowercase words for keyword search: 'Pre-existing' -> ['pre', 'existing']."""
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self):
        with CHUNKS_FILE.open(encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.plans = sorted({c["plan"] for c in self.chunks})
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.model = SentenceTransformer(MODEL_NAME)
        self.collection = chromadb.PersistentClient(path=str(INDEX_DIR)).get_collection("policies")

    def search(self, question, k=5, pool=20, plan=None):
        """Return the k best chunks. Pass plan="..." to search only one policy."""
        candidates = [i for i, c in enumerate(self.chunks) if plan is None or c["plan"] == plan]
        if not candidates:
            return []
        pool = min(pool, len(candidates))

        # 1. Meaning search: closest embeddings (optionally only within one plan)
        q = self.model.encode(QUERY_PREFIX + question, normalize_embeddings=True)
        vector_ids = self.collection.query(
            query_embeddings=[q.tolist()],
            n_results=pool,
            where={"plan": plan} if plan else None,
        )["ids"][0]

        # 2. Keyword search: BM25 scores for the candidate chunks, keep the best `pool`
        scores = self.bm25.get_scores(tokenize(question))
        best = sorted(candidates, key=lambda i: scores[i], reverse=True)[:pool]
        keyword_ids = [self.chunks[i]["chunk_id"] for i in best]

        # 3. Reciprocal rank fusion: being near the top of either list scores well
        fused = {}
        for ids in (vector_ids, keyword_ids):
            for rank, chunk_id in enumerate(ids):
                fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (60 + rank)

        top_ids = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self.by_id[i] for i in top_ids]

    def search_each_plan(self, question, k=3):
        """Return the k best chunks from EVERY policy, grouped policy by policy.

        Used for "All policies": the same question usually has a different answer
        in each policy, so each one gets its own clauses instead of competing.
        """
        return [chunk for plan in self.plans for chunk in self.search(question, k=k, plan=plan)]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is the waiting period for pre-existing diseases?"
    for chunk in Retriever().search(question):
        print(f"--- page {chunk['printed_page']} ({chunk['chunk_id']})")
        print(chunk["text"][:300], "\n")