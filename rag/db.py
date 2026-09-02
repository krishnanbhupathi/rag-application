import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .config import DB_URL, EMBED_DIM, PROJECT_ROOT
from .models import Chunk, RawDocument

_pool: ConnectionPool | None = None


def init_schema() -> None:
    """Run the DDL on a plain connection. Must happen before the pool opens:
    register_vector() fails if the `vector` extension doesn't exist yet."""
    ddl = (PROJECT_ROOT / "schema.sql").read_text()
    with psycopg.connect(DB_URL, autocommit=True) as conn:
        conn.execute(ddl)
        row = conn.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
        ).fetchone()
        if row and row[0] != EMBED_DIM:
            raise RuntimeError(
                f"schema.sql declares vector({row[0]}), config says {EMBED_DIM}"
            )


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        init_schema()
        _pool = ConnectionPool(
            DB_URL, min_size=1, max_size=5,
            configure=register_vector, open=True,
        )
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def get_document_hash(conn, source_uri: str) -> str | None:
    row = conn.execute(
        "SELECT content_hash FROM documents WHERE source_uri = %s", (source_uri,)
    ).fetchone()
    return row[0] if row else None


def upsert_document(conn, doc: RawDocument) -> str:
    row = conn.execute(
        """
        INSERT INTO documents (source_type, source_uri, title, content_hash, metadata)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (source_uri) DO UPDATE
            SET title     = EXCLUDED.title,
                content_hash  = EXCLUDED.content_hash,
                metadata      = EXCLUDED.metadata,
                ingested_at   = now()
        RETURNING id
        """,
        (doc.source_type.value, doc.source_uri, doc.title,
         doc.content_hash, Jsonb(doc.metadata)),
    ).fetchone()
    return str(row[0])


def replace_chunks(conn, document_id: str, chunks: list[Chunk],
                   embeddings: list[np.ndarray]) -> int:
    conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
    rows = [
        (document_id, c.chunk_index, c.content, c.token_count, e, Jsonb(c.metadata))
        for c, e in zip(chunks, embeddings, strict=True)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (document_id, chunk_index, content, token_count,"
            " embedding, metadata) VALUES (%s, %s, %s, %s, %s, %s)",
            rows,
        )
    return len(rows)
