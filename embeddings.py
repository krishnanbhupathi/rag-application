from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import EMBED_BATCH_SIZE, EMBED_DIM, EMBED_MODEL, QUERY_PREFIX


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    model = SentenceTransformer(EMBED_MODEL, device=_device())
    # sentence-transformers renamed this method; support both spellings.
    probe = getattr(model, "get_embedding_dimension", None) \
            or model.get_sentence_embedding_dimension
    actual = probe()
    if actual != EMBED_DIM:
        raise RuntimeError(f"EMBED_DIM={EMBED_DIM} but {EMBED_MODEL} emits {actual}")
    return model


def embed_documents(texts: list[str]) -> list[np.ndarray]:
    """Passages. NO prefix - BGE is asymmetric."""
    if not texts:
        return []
    vecs = get_model().encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 100,
    )
    return [v.astype(np.float32) for v in vecs]


def embed_query(text: str) -> np.ndarray:
    """Queries. WITH prefix - BGE is asymmetric."""
    vec = get_model().encode(
        [QUERY_PREFIX + text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    return vec.astype(np.float32)
