import hashlib
import re
from pathlib import Path

import pymupdf

from models import Page, RawDocument, SourceType

_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl",
}

_HYPHEN_BREAK = re.compile(r"([A-Za-z]{2,})-\n\s*([A-Za-z]{2,})")


def compute_content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_body_block(b: tuple) -> bool:
    """Keep real body text. Drop image blocks, rotated margin stamps
    (arXiv's vertical banner), and bare page-number footers."""
    x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
    text = b[4].strip()
    if b[6] != 0 or not text:                    # block_type 0 == text
        return False
    w, h = x1 - x0, y1 - y0
    if h > 0 and w / h < 0.15:                   # tall + narrow ⇒ rotated
        return False
    if re.fullmatch(r"\d{1,4}", text):           # lone page number
        return False
    return True


def _normalize(text: str) -> str:
    """Ligatures and soft hyphens. Runs BEFORE the vocabulary is built so that
    the vocabulary and the de-hyphenation step see identical spellings."""
    for lig, repl in _LIGATURES.items():
        text = text.replace(lig, repl)
    return text.replace("\u00ad", "")


def _document_vocabulary(texts: list[str]) -> set[str]:
    """Every word in the document, minus the hyphen-break sites themselves —
    otherwise fragments like 'retrie' would be learned as real words."""
    vocab: set[str] = set()
    for t in texts:
        t = _HYPHEN_BREAK.sub(" ", t)
        vocab.update(w.lower() for w in re.findall(r"[A-Za-z]{2,}", t))
    return vocab


def dehyphenate(text: str, vocab: set[str]) -> str:
    """A line-end hyphen is ambiguous: soft (a broken word) or hard (a real
    compound). Use the document itself as the dictionary."""
    def choose(m: re.Match) -> str:
        a, b = m.group(1), m.group(2)
        if (a + b).lower() in vocab:      # seen whole elsewhere → soft hyphen
            return a + b
        if a.lower() in vocab:            # 'a' is a complete word → compound
            return f"{a}-{b}"
        return a + b                      # 'a' is a fragment → broken word
    return _HYPHEN_BREAK.sub(choose, text)


def clean_pdf_text(text: str, vocab: set[str]) -> str:
    text = dehyphenate(text, vocab)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)   # unwrap lines, keep paragraphs
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> RawDocument:
    data = path.read_bytes()
    pdf = pymupdf.open(stream=data, filetype="pdf")

    # Pass 1 — extract and normalize, no de-hyphenation yet.
    raw: list[tuple[int, str]] = []
    for i, page in enumerate(pdf, start=1):
        blocks = page.get_text("blocks", sort=True)
        kept = [_normalize(b[4].strip()) for b in blocks if _is_body_block(b)]
        if kept:
            raw.append((i, "\n\n".join(kept)))

    # Pass 2 — learn the document's vocabulary, then clean with it.
    vocab = _document_vocabulary([t for _, t in raw])
    pages = [Page(page_number=i, text=clean_pdf_text(t, vocab)) for i, t in raw]
    pages = [p for p in pages if p.text]

    meta = pdf.metadata or {}
    title = (meta.get("title") or "").strip() or path.stem
    doc_meta = {"page_count": pdf.page_count, "producer": meta.get("producer")}
    pdf.close()

    return RawDocument(
        source_type=SourceType.PDF,
        source_uri=str(path.resolve()),
        title=title,
        pages=pages,
        content_hash=compute_content_hash(data),
        metadata=doc_meta,
    )