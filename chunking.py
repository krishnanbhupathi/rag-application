from functools import lru_cache

from transformers import AutoTokenizer

from config import CHUNK_OVERLAP_TOKENS, CHUNK_TOKENS, EMBED_MODEL, MIN_CHUNK_CHARS
from models import RawDocument, Chunk

SEPARATORS = ["\n\n", "\n", ". ", " "]


@lru_cache(maxsize=1)
def _tokenizer():
    return AutoTokenizer.from_pretrained(EMBED_MODEL)


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Last resort: no separator worked, slice on token boundaries."""
    tok = _tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    return [tok.decode(ids[i:i + max_tokens]) for i in range(0, len(ids), max_tokens)]


def _atomize(text: str, seps: list[str], max_tokens: int) -> list[str]:
    """Break text into pieces that each fit the budget, cutting at the
    largest structural separator that works."""
    if count_tokens(text) <= max_tokens:
        return [text]
    if not seps:
        return _hard_split(text, max_tokens)

    sep, rest = seps[0], seps[1:]
    out: list[str] = []
    parts = text.split(sep)
    for j, part in enumerate(parts):
        if not part.strip():
            continue
        piece = part + (sep if j < len(parts) - 1 else "")
        if count_tokens(piece) <= max_tokens:
            out.append(piece)
        else:
            out.extend(_atomize(piece, rest, max_tokens))
    return out


def _pack(pieces: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Greedily fill chunks to the budget, carrying an overlap tail forward."""
    sized = [(p, count_tokens(p)) for p in pieces]   #count once, not repeatedly
    chunks: list[str] = []
    cur: list[tuple[str, int]] = []
    cur_tokens = 0

    for piece, n in sized:
        if cur and cur_tokens + n > max_tokens:
            chunks.append("".join(p for p, _ in cur).strip())
            tail: list[tuple[str, int]] = []
            tail_tokens = 0
            for prev, pn in reversed(cur):          #walk backwards for the overlap
                if tail_tokens + pn > overlap_tokens:
                    break
                tail.insert(0, (prev, pn))
                tail_tokens += pn
            while tail and tail_tokens + n > max_tokens:
                _, dropped = tail.pop(0)
                tail_tokens -= dropped
            cur, cur_tokens = tail, tail_tokens
        cur.append((piece, n))
        cur_tokens += n

    if cur:
        chunks.append("".join(p for p, _ in cur).strip())
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def split_text(text: str, *, max_tokens: int = CHUNK_TOKENS,
               overlap_tokens: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    return _pack(_atomize(text, SEPARATORS, max_tokens), max_tokens, overlap_tokens)


def chunk_document(doc: RawDocument, *, max_tokens: int = CHUNK_TOKENS,
                   overlap_tokens: int = CHUNK_OVERLAP_TOKENS) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for page in doc.pages:
        for text in split_text(page.text, max_tokens=max_tokens,
                               overlap_tokens=overlap_tokens):
            chunks.append(Chunk(
                chunk_index=idx,
                content=text,
                token_count=count_tokens(text),
                metadata={"page": page.page_number, "title": doc.title},
            ))
            idx += 1
    return chunks
