import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# rag/config.py is one level down now, so climb twice to reach the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")

# --- embeddings -------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
EMBED_BATCH_SIZE = 64

# BGE is ASYMMETRIC: queries get this prefix, documents get nothing.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- chunking ----------------------------------------------------------------
# bge-small truncates at 512 tokens. Anything past that is SILENTLY DISCARDED.
# 400 leaves headroom so overlap-packing never crosses the cliff.
CHUNK_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 60
MIN_CHUNK_CHARS = 50
