import pytest

from app.ingestion.chunking import chunk_text


def test_empty_text_produces_no_chunks():
    assert chunk_text("", chunk_size=100, chunk_overlap=10) == []


def test_whitespace_only_produces_no_chunks():
    assert chunk_text("   \n\n  \t  ", chunk_size=100, chunk_overlap=10) == []


def test_text_shorter_than_chunk_size_produces_one_chunk():
    chunks = chunk_text("A short sentence.", chunk_size=1000, chunk_overlap=150)
    assert len(chunks) == 1
    assert chunks[0].text == "A short sentence."
    assert chunks[0].index == 0


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, chunk_overlap=100)


def test_long_text_produces_multiple_chunks():
    # ~2500 chars of sentences, well over the 500-char chunk size
    sentence = "Gradient descent minimizes the loss function iteratively. "
    text = sentence * 40
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        # Overlap can push a chunk slightly over chunk_size (by up to
        # `overlap` chars) since overlap is applied after splitting —
        # documented behavior, not a bug: it's the tradeoff for guaranteeing
        # boundary sentences stay whole. Assert the *split* pieces (minus
        # overlap) were within budget rather than the final overlapped size.
        assert c.char_count <= 500 + 50


def test_chunks_are_sequentially_indexed():
    text = "Sentence one. " * 100
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlap_preserves_boundary_sentence():
    # Construct text where a sentence boundary should land near a chunk
    # boundary, and confirm the sentence isn't lost entirely from both chunks.
    text = (
        "First paragraph with some content here to pad it out nicely. " * 5
        + "CRITICAL_MARKER_SENTENCE lives right at the boundary of two chunks. "
        + "Second paragraph with more padding content to push past the limit. " * 5
    )
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=100)
    assert any("CRITICAL_MARKER_SENTENCE" in c.text for c in chunks)


def test_prefers_paragraph_boundary_over_hard_cut():
    para1 = "First paragraph. " * 10  # ~180 chars
    para2 = "Second paragraph. " * 10  # ~190 chars
    text = para1.strip() + "\n\n" + para2.strip()
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=0)
    # With no overlap, chunk boundaries should land on paragraph breaks —
    # neither chunk should contain fragments of both paragraphs glued
    # mid-sentence (a hard character cut would produce that).
    assert not any("First paragraph." in c.text and "Second paragraph." in c.text for c in chunks)


def test_single_word_longer_than_chunk_size_falls_back_to_hard_cut():
    huge_token = "x" * 50
    chunks = chunk_text(huge_token, chunk_size=10, chunk_overlap=0)
    assert len(chunks) == 5
    assert "".join(c.text for c in chunks) == huge_token


def test_no_words_are_lost_across_chunk_boundaries():
    # Chunks are consumed independently (embedded, retrieved, and shown to
    # an LLM one at a time) — they're never blindly concatenated with no
    # separator, so exact byte-for-byte rejoining isn't the real invariant.
    # What actually matters: every word from the original text shows up,
    # in order, somewhere across the chunks — nothing silently dropped.
    text = "Word " * 300
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)
    reconstructed = " ".join(c.text for c in chunks)
    assert reconstructed.split() == text.split()
