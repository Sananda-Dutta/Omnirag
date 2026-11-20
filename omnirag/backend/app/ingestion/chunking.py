"""
Text chunking.

Why chunks exist at all: embedding models and LLM context windows both have
input limits, and retrieval precision degrades as chunk size grows — a
10,000-character chunk containing the right answer buried in unrelated
content embeds to a vector that's a blurry average of everything in it,
making it harder for a query embedding to match closely. Smaller, focused
chunks retrieve more precisely. The tradeoff: too small and a chunk loses
the surrounding context needed to make sense of it on its own (a chunk that's
just "the accuracy was 94.2%" is useless without knowing which model/dataset).

Why ~1000 characters (roughly 150-250 tokens) as the default: small enough
for precise retrieval, large enough to usually contain a complete thought.
This is a starting point, not a law — Phase 15 (RAG evaluation) is where we
actually measure retrieval quality and can justify tuning it with data
instead of intuition.

Why overlap (default 150 chars): without it, a sentence that happens to
fall across a chunk boundary gets split, and neither resulting chunk
contains the complete thought — the embedding for both halves is degraded,
and retrieval can miss the sentence entirely. Overlap means the boundary
sentence is intact in at least one of the two chunks.

Why recursive/boundary-aware splitting instead of blind `text[i:i+1000]`
slicing: a hard character cut can slice through the middle of a word, which
both looks broken to a human reading a citation and slightly degrades the
embedding (a truncated word is a different token sequence than the whole
word). This splitter tries paragraph breaks first, then sentence breaks,
then word breaks, and only falls back to a hard character cut if a single
"word" (e.g. a long URL or hash) is itself longer than the chunk size.
"""

import re
from dataclasses import dataclass

# Ordered from "most preferred split point" to "last resort." Trying these
# in order is what makes this "recursive" — split on the first separator
# that actually breaks the text into pieces, recursing into any piece
# that's still too big using the next separator down the list.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    char_count: int
    page_number: int | None = None


def _split_on_separator(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)  # last-resort: split into individual characters
    parts = text.split(separator)
    # Re-attach the separator to all but the last piece, so rejoining pieces
    # later reproduces the original text exactly (important for sentence
    # separators like ". " — dropping it would glue sentences together).
    return [p + separator for p in parts[:-1]] + parts[-1:]


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        # No separator left to try — hard cut. Only reached for pathological
        # input (e.g. one 5000-character "word" with no spaces at all).
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, *rest = separators
    pieces = _split_on_separator(text, separator)

    # Greedily pack pieces into chunks up to chunk_size, recursing into any
    # individual piece that's still too big on its own.
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(piece) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_recursive_split(piece, chunk_size, rest))
            continue

        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current:
                chunks.append(current)
            current = piece

    if current:
        chunks.append(current)

    return chunks


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for prev, current in zip(chunks, chunks[1:]):
        tail = prev[-overlap:]
        result.append(tail + current)
    return result


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """
    Splits `text` into overlapping chunks, preferring paragraph/sentence/word
    boundaries over hard character cuts. Returns [] for empty/whitespace-only
    input rather than a single empty chunk.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = re.sub(r"[ \t]+", " ", text).strip()
    if not normalized:
        return []

    raw_pieces = _recursive_split(normalized, chunk_size, _SEPARATORS)
    raw_pieces = [p.strip() for p in raw_pieces if p.strip()]
    overlapped = _apply_overlap(raw_pieces, chunk_overlap)

    return [
        Chunk(index=i, text=piece, char_count=len(piece)) for i, piece in enumerate(overlapped)
    ]


def chunk_pages(pages: list[str], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """
    Page-aware chunking (Phase 9), used for PDFs so every chunk can carry an
    accurate `page_number` for citations.

    Deliberately chunks each page independently — via repeated chunk_text()
    calls, not a single call across the whole joined document — rather than
    chunking the full joined text and trying to map each resulting chunk's
    character offsets back to a source page after the fact. The offset-
    mapping approach is more precise in principle (a chunk could legitimately
    span a page break) but requires chunk_text() to track provenance through
    every split/overlap step, which the existing (already well-tested)
    implementation doesn't do and wasn't designed to.

    The real tradeoff, stated plainly: a paragraph that spans a page break
    gets split at that page boundary even if it wouldn't otherwise hit
    chunk_size — trading a small amount of chunk continuity for citations
    that are always unambiguous about which single page they came from. For
    a citation system, "always know which page" matters more than "never
    split a mid-page-break paragraph."

    Chunk `index` is renumbered to be sequential across the whole document
    (not reset per page) — chunk_index is meant to identify a chunk's
    position in the document overall, not its position within its page.
    """
    all_chunks: list[Chunk] = []
    for page_num, page_text in enumerate(pages, start=1):
        page_chunks = chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for c in page_chunks:
            all_chunks.append(
                Chunk(
                    index=len(all_chunks),
                    text=c.text,
                    char_count=c.char_count,
                    page_number=page_num,
                )
            )
    return all_chunks
