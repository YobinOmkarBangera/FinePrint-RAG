import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_DIR / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = PROJECT_DIR / "data" / "index"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION = "policies"


def load_chunks():
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    chunks = load_chunks()
    model = SentenceTransformer(MODEL_NAME)

    # normalize_embeddings=True makes the default distance rank results like cosine similarity
    embeddings = model.encode(
        [c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True
    )

    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    # Rebuild from scratch each time so deleted or renamed chunks don't linger
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[{
            "insurer": c["insurer"],
            "plan": c["plan"],
            "page": c["page"],
            "printed_page": c["printed_page"] or 0,  # Chroma doesn't allow None; 0 = cover page
        } for c in chunks],
    )
    print(f"Indexed {collection.count()} chunks into {INDEX_DIR}")


if __name__ == "__main__":
    main()