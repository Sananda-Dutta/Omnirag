"""
Reranker abstraction + a real local lexical implementation.

Why a separate reranking stage after fusion at all: RRF (fusion.py) merges
two independently-ranked lists using only rank position — it never looks at
actual text, so it can't distinguish "this candidate barely matches" from
"this candidate matches extremely well" within a single list; everything
collapses to its rank. A reranking pass that actually looks at
query-vs-candidate content recovers that lost signal before the list is
truncated to RAG_TOP_K.

What's implemented here: a real, working lexical reranker — term overlap
between the query and each candidate, weighted by normalized term frequency
and an in-context inverse-document-frequency approximation (how many of
*these* candidates contain each shared term, not a corpus-wide statistic —
there's no separately maintained corpus-frequency index in this project).
It is explicitly NOT a cross-encoder: it can't judge semantic equivalence
between differently-worded but same-meaning text the way a trained neural
reranker (Cohere Rerank, a sentence-transformers cross-encoder) can. No
such model is reachable from this sandbox (no network access to Cohere's
API, no HuggingFace access to download a cross-encoder) — consistent with
LocalHashingEmbeddingProvider and LocalExtractiveLLMProvider elsewhere in
this project, the honest choice is a real, working, clearly-labeled
non-neural technique instead of faking one or silently skipping the stage.
"""

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from uuid import UUID

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass(frozen=True)
class RerankCandidate:
    chunk_id: UUID
    text: str


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[RerankCandidate]) -> list[tuple[UUID, float]]:
        """Returns (chunk_id, score) pairs reordered best-first. Scores are
        returned (not just the reordering) so a caller displaying results
        can show the score that actually determined the final order,
        instead of a stale pre-rerank score."""
        ...


class LocalLexicalReranker(Reranker):
    def rerank(self, query: str, candidates: list[RerankCandidate]) -> list[tuple[UUID, float]]:
        if not candidates:
            return []

        query_terms = set(_tokenize(query))
        if not query_terms:
            return [(c.chunk_id, 0.0) for c in candidates]

        tokenized = [_tokenize(c.text) for c in candidates]
        term_sets = [set(tokens) for tokens in tokenized]

        # In-context document frequency: how many of THESE candidates
        # contain each query term. A term present in only one or two
        # candidates is more discriminating than one present in nearly all
        # of them — the same intuition as corpus-wide IDF, computed over
        # just the small candidate set actually in play.
        doc_freq: Counter[str] = Counter()
        for terms in term_sets:
            for t in terms & query_terms:
                doc_freq[t] += 1

        scored: list[tuple[float, UUID]] = []
        for candidate, tokens, terms in zip(candidates, tokenized, term_sets):
            if not tokens:
                scored.append((0.0, candidate.chunk_id))
                continue
            tf = Counter(tokens)
            score = sum(
                (tf[t] / len(tokens)) * (1.0 / (doc_freq[t] + 1))  # +1 smoothing: avoids
                # div-by-zero and keeps a term present in every candidate from
                # scoring exactly zero, which would make it indistinguishable
                # from "term absent entirely."
                for t in terms & query_terms
            )
            scored.append((score, candidate.chunk_id))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [(chunk_id, score) for score, chunk_id in scored]
