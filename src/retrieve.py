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
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.model = SentenceTransformer(MODEL_NAME)
        self.collection = chromadb.PersistentClient(path=str(INDEX_DIR)).get_collection("policies")

    def search(self, question, k=5, pool=20):
        pool = min(pool, len(self.chunks))

        # 1. Meaning search: closest embeddings
        q = self.model.encode(QUERY_PREFIX + question, normalize_embeddings=True)
        vector_ids = self.collection.query(query_embeddings=[q.tolist()], n_results=pool)["ids"][0]

        # 2. Keyword search: BM25 scores for every chunk, keep the best `pool`
        scores = self.bm25.get_scores(tokenize(question))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:pool]
        keyword_ids = [self.chunks[i]["chunk_id"] for i in best]

        # 3. Reciprocal rank fusion: being near the top of either list scores well
        fused = {}
        for ids in (vector_ids, keyword_ids):
            for rank, chunk_id in enumerate(ids):
                fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (60 + rank)

        top_ids = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self.by_id[i] for i in top_ids]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is the waiting period for pre-existing diseases?"
    for chunk in Retriever().search(question):
        print(f"--- page {chunk['printed_page']} ({chunk['chunk_id']})")
        print(chunk["text"][:300], "\n")