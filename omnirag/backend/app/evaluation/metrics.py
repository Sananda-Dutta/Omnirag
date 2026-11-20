"""
Retrieval metrics for Phase 15.

Deliberately plain functions, not part of the EvaluationProvider Strategy
hierarchy (see base.py's docstring for why generation scoring needs that
abstraction and this doesn't): given a list of retrieved document
identifiers and a set of expected ones, precision/recall/MRR/hit-rate are
unambiguous set-and-order arithmetic. There's exactly one correct answer
for a given input, so there's nothing here to swap out — no provider, no
config, just math, tested directly.

Every function takes `retrieved: list[str]` (ranked, best-first) and
`expected: set[str]` (unordered — "any of these would be a correct
source"), using whatever identifier the caller considers a document's
identity (its filename in practice — see evaluation_service.py). Order in
`retrieved` matters for MRR; order in `expected` never matters.
"""


def precision_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """Of the top-k retrieved documents, what fraction are actually
    relevant? Undefined (returns 0.0, not an error) when retrieved is
    shorter than k or empty — a RAG pipeline that retrieves nothing scores
    zero precision, not "not applicable"."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in expected)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """Of all the documents that were actually relevant, what fraction did
    the top-k retrieval surface? Returns 1.0 when `expected` is empty —
    vacuously true (nothing needed to be found, and nothing was missed),
    not 0.0, since 0.0 would penalize a question with no labeled expected
    sources rather than treat it as not applicable to this metric."""
    if not expected:
        return 1.0
    top_k = set(retrieved[:k])
    hits = len(top_k & expected)
    return hits / len(expected)


def mean_reciprocal_rank(retrieved: list[str], expected: set[str]) -> float:
    """1 / (rank of the first relevant result), 0.0 if none of the
    retrieved documents are relevant. Rewards getting *a* relevant result
    near the top, regardless of how many more relevant ones exist further
    down — the metric a "did the answer's top citation actually come from
    the right place" question wants, as distinct from precision/recall
    which treat every position equally."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


def hit_rate(retrieved: list[str], expected: set[str], k: int) -> float:
    """1.0 if at least one of the top-k retrieved documents is relevant,
    else 0.0. The coarsest, most forgiving retrieval metric — "did we get
    *anything* right" — useful as a sanity floor alongside the stricter
    precision/recall numbers, which can both be low even when the top
    result was actually the right one."""
    return 1.0 if set(retrieved[:k]) & expected else 0.0
