from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    PDF = "pdf"
    URL = "url"
    MARKDOWN = "markdown"


class Page(BaseModel):
    page_number: int           # 1-based  - matches how a human cites a PDF
    text: str


class RawDocument(BaseModel):
    source_type: SourceType
    source_uri: str
    title: str | None = None
    pages: list[Page]
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_index: int
    content: str
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResult(BaseModel):
    document_id: str
    source_uri: str
    chunks_written: int
    skipped_unchanged: bool
    duration_ms: int