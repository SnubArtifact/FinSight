import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_api_key")

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")
LLM_MODEL = os.getenv("GEMINI_CHAT_MODEL", "models/gemini-2.5-flash")

VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store")
VECTOR_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss.index")
VECTOR_META_PATH = os.path.join(VECTOR_STORE_DIR, "chunks.json")
