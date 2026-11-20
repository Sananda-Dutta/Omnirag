"""
Local hashing-based embedding provider.

What this is: the "hashing trick" (Weinberger et al., 2009 — also what
scikit-learn's HashingVectorizer implements). Each word (and word-bigram, for
a little word-order sensitivity) is hashed into one of `dimension` buckets,
with a random +1/-1 sign attached, and a chunk's vector is the sum of its
tokens' contributions, L2-normalized at the end. Two chunks sharing more
vocabulary land closer together in cosine similarity.

Why this is the DEFAULT provider in this repo (not just a fallback): it
needs no API key, no network call, and no downloaded model weights — which
matters concretely here, since this sandboxed environment has no egress to
OpenAI or HuggingFace at all. Using this as the default is what makes the
full ingestion -> chunking -> embedding pipeline something this repo can
actually run and test end-to-end, rather than a pipeline that only compiles.

What this is NOT: a substitute for a real neural embedding model. It has no
notion of synonymy or semantics — "car" and "automobile" hash to unrelated
buckets and look completely dissimilar, where a real model like
text-embedding-3-small or an open sentence-transformer would place them
close together. It's a legitimate, real, non-fake algorithm (this is a
genuine technique, not a placeholder pretending to be one) — but its
retrieval quality ceiling is bag-of-words, not semantic search. Swapping in
`OpenAIEmbeddingProvider` (or a self-hosted sentence-transformers provider,
a natural addition later) via `EMBEDDING_PROVIDER=openai` is how this
project would actually get semantic retrieval quality in a real deployment.
"""

import hashlib
import math
import re

from app.embeddings.base import EmbeddingProvider

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    words = _WORD_RE.findall(text.lower())
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def _hash_token(token: str, dimension: int) -> tuple[int, float]:
    # hashlib, not Python's built-in hash(): str hashing is randomized per
    # process (PYTHONHASHSEED) unless disabled, which would make embeddings
    # of the same text different across runs/processes — unusable for
    # anything that compares vectors computed at different times (exactly
    # what retrieval does: embed a query now, compare against chunks
    # embedded during ingestion, possibly in a different process).
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % dimension
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return index, sign


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


class LocalHashingEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return f"local-hashing-{self._dimension}d"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # No actual I/O, so no real batching benefit — async signature kept
        # only so this is interchangeable with providers that do call out
        # over the network, without callers needing to know the difference.
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _tokenize(text):
            index, sign = _hash_token(token, self._dimension)
            vector[index] += sign
        return _l2_normalize(vector)
