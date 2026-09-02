import argparse
import time
from pathlib import Path

from chunking import chunk_document
from db import get_document_hash, get_pool, replace_chunks, upsert_document, close_pool
from embeddings import embed_documents
from loaders import load_pdf
from models import IngestResult


def ingest_pdf(path: Path, *, force: bool = False) -> IngestResult:
    t0 = time.perf_counter()
    doc = load_pdf(path)
    pool = get_pool()

    # 1. Cheap short-circuit on an unchanged file.
    with pool.connection() as conn:
        if not force and get_document_hash(conn, doc.source_uri) == doc.content_hash:
            doc_id = conn.execute(
                "SELECT id FROM documents WHERE source_uri = %s", (doc.source_uri,)
            ).fetchone()[0]
            return IngestResult(
                document_id=str(doc_id), source_uri=doc.source_uri,
                chunks_written=0, skipped_unchanged=True,
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )

    # 2. Chunk + embed with NO connection held - this is the slow part
    chunks = chunk_document(doc)
    vectors = embed_documents([c.content for c in chunks])

    # 3. One short write transaction.
    with pool.connection() as conn, conn.transaction():
        doc_id = upsert_document(conn, doc)
        written = replace_chunks(conn, doc_id, chunks, vectors)

    return IngestResult(
        document_id=doc_id, source_uri=doc.source_uri,
        chunks_written=written, skipped_unchanged=False,
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dump-text", type=Path,
                    help="Write extracted text here and exit. USE THIS FIRST.")
    args = ap.parse_args()

    try:
        if args.dump_text:
            doc = load_pdf(args.path)
            args.dump_text.write_text(
                "\n\n".join(f"===== page {p.page_number} =====\n{p.text}"
                            for p in doc.pages)
            )
            print(f"wrote {args.dump_text} ({len(doc.pages)} pages)")
            return

        print(ingest_pdf(args.path, force=args.force).model_dump_json(indent=2))
    finally:
        close_pool()


if __name__ == "__main__":
    main()