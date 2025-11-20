import json
import uuid
from pathlib import Path

import faiss
import google.generativeai as genai
import numpy as np

from utils.config import (
    EMBED_MODEL,
    GEMINI_API_KEY,
    VECTOR_INDEX_PATH,
    VECTOR_META_PATH,
)

genai.configure(api_key=GEMINI_API_KEY)


def embed_chunks(chunks: list):
    """Create embeddings for chunks."""
    vectors = []
    for chunk in chunks:
        response = genai.embed_content(model=EMBED_MODEL, content=chunk)
        vectors.append(response["embedding"])
    return vectors


def store_chunks(chunks: list, embeddings: list):
    """Persist embeddings locally using FAISS."""
    if not chunks:
        print("[!] No chunks to store.")
        return
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings length mismatch.")

    vectors = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)
    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    store_dir = Path(VECTOR_INDEX_PATH).parent
    store_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, VECTOR_INDEX_PATH)

    metadata = [
        {"id": str(uuid.uuid4()), "text": chunk}
        for chunk in chunks
    ]
    with open(VECTOR_META_PATH, "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, ensure_ascii=False, indent=2)

    print(f"[✓] Stored {len(metadata)} chunks in local FAISS index.")
