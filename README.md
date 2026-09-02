# Multi-Source RAG Service

A RAG service over PDFs, web pages and Markdown that cites its sources, refuses
when the retrieved context doesn't support an answer, and **reports measured
retrieval and answer quality**.

> **Status: in progress — Day 1 of 7.** PDF ingestion, chunking, embedding and
> pgvector storage are working. Retrieval, grounding and evaluation are not built
> yet. No quality numbers are claimed until `eval/run_eval.py` produces them.

## Why this isn't another "chat with your PDF"

Three things, built deliberately:

1. **Hybrid retrieval + rerank** — dense (pgvector) and sparse (Postgres FTS)
   fused with Reciprocal Rank Fusion, then a local cross-encoder reranks. Not
   cosine similarity alone.
2. **Grounding guardrail** — below a context-support threshold it refuses rather
   than answers. Every answer cites the chunks it used.
3. **Measured evaluation** — a hand-labelled Q/A set scored for faithfulness,
   context-recall and answer-relevance, with the numbers in this README.

No LangChain or LlamaIndex. The retrieval loop is written from primitives.

## Architecture

```
PDF/URL/MD ──► loaders ──► chunking ──► embeddings ──► Postgres
                                                       ├── vector(384)  HNSW
                                                       └── tsvector     GIN
```

| Component | File | Status |
|---|---|---|
| PDF loader (PyMuPDF, layout-aware) | `loaders.py` | ✅ |
| Token-budgeted recursive chunker | `chunking.py` | ✅ |
| Local embeddings (bge-small-en-v1.5) | `embeddings.py` | ✅ |
| Postgres + pgvector schema | `schema.sql`, `db.py` | ✅ |
| URL + Markdown loaders | `loaders.py` | ⬜ Day 2 |
| Hybrid retrieval + RRF + rerank | `retrieve.py` | ⬜ Day 3 |
| Grounded answers + refusal | `answer.py` | ⬜ Day 4 |
| Self-correction retry, tracing, API | `correct.py`, `api.py` | ⬜ Day 5 |
| Eval set + metrics | `eval/` | ⬜ Day 6 |

## Quickstart

```bash
docker compose up -d
pip install -r requirements.txt
python ingest.py corpus/yourfile.pdf
```

Inspect what the PDF extractor actually produced before trusting any chunk:

```bash
python ingest.py corpus/yourfile.pdf --dump-text /tmp/extracted.txt
```

## Design decisions

**Local `bge-small-en-v1.5` (384d) over a hosted embedding API.** With a
cross-encoder reranker downstream, the bi-encoder only needs good recall@50 —
precision at the top is the reranker's job. That narrows the quality gap enough
that reproducible, zero-cost, offline embeddings win. Re-embedding the corpus
while tuning the chunker is free, so the experiment actually gets run.

**Chunk budget of 400 tokens, not 512.** `bge-small` truncates at 512 and
discards the remainder silently. Token counts use the embedding model's own
tokenizer rather than an approximation, so the budget is enforced in the units
the model actually truncates in.

**pgvector rather than a dedicated vector database.** Vectors live in ordinary
rows, so hybrid retrieval is one SQL query over one table instead of two systems
kept in sync — and metadata filters are just `WHERE`. The tradeoff is weaker
behaviour at very large scale and index builds competing with the same instance.

**Chunk per page.** Preserves the page number needed for citations, at the cost
of orphaning sections that straddle a page break (see Limitations).

**Document-vocabulary de-hyphenation.** A line-end hyphen is ambiguous: soft
(`retrie-\nval`) or hard (`sequence-\naligned`). The document itself is used as
the dictionary. The default favours keeping the hyphen, because a wrong join
produces an unsearchable token while a wrong split leaves two searchable words.

## Known limitations

- **Page-boundary chunking** orphans sections that straddle a page break; overlap
  cannot cross pages. Observed: a 48-token chunk containing a section heading and
  one sentence.
- **Front matter** (title, authors, licence boilerplate) is not stripped and
  occupies part of the first, highest-value chunk.
- **Tables are flattened** by text extraction — row/column association is lost.
  Table-lookup questions are excluded from the eval set for this reason.
- **The ingest cache keys on file bytes only.** Changing the embedding model or
  chunk config does not invalidate it. A same-dimension model swap would leave
  stale vectors in a second embedding space with no error raised.

## Evaluation

Not yet measured. Numbers will be filled in from `eval/run_eval.py` output on
Day 6 — none are claimed before then.