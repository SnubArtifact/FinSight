import json
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


def _load_index_and_metadata():
    index_path = Path(VECTOR_INDEX_PATH)
    meta_path = Path(VECTOR_META_PATH)
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Vector store not found. Run ingestion to create the FAISS index."
        )

    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    return index, metadata


def retrieve(query: str, top_k: int = 5, conversation_context: str = None):
    """Retrieve top-k chunk texts.
    
    Args:
        query: User's current question
        top_k: Number of chunks to retrieve
        conversation_context: Optional context from previous conversation to improve retrieval
    """
    index, metadata = _load_index_and_metadata()
    if not metadata:
        return []

    # Enhance query with conversation context for better retrieval of follow-ups
    enhanced_query = query
    if conversation_context:
        enhanced_query = f"{query} {conversation_context}"

    # Embed query
    q_emb = genai.embed_content(
        model=EMBED_MODEL,
        content=enhanced_query
    )["embedding"]
    query_vec = np.array([q_emb], dtype="float32")
    faiss.normalize_L2(query_vec)

    limit = min(top_k, len(metadata))
    scores, neighbors = index.search(query_vec, limit)

    results = []
    for idx in neighbors[0]:
        if idx == -1:
            continue
        results.append(metadata[idx]["text"])
    return results
