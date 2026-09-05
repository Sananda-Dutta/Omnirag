"""
Reciprocal Rank Fusion (RRF).

Why rank position instead of raw scores: Qdrant's cosine similarity scores
and Postgres's ts_rank_cd scores live on completely different, incomparable
scales (roughly 0-1 for cosine similarity; an unbounded, corpus-dependent
value for ts_rank_cd). Averaging or summing them directly would be
meaningless — a 0.8 from one method and a 0.8 from the other don't
represent comparable relevance. RRF sidesteps the normalization problem
entirely by using each item's *rank position* within its own list, which is
always comparable regardless of the underlying scoring method.

Formula: for each item, RRF_score = sum over every list it appears in of
1 / (k + rank), where rank is 1-indexed position and k is a damping
constant. An item appearing (even at a middling rank) in both the dense and
keyword result lists accumulates score from both — which is exactly the
"complementary recall" hybrid retrieval is for: a chunk both methods agree
on outranks a chunk only one method found, even if that one method ranked
it #1.

k=60 (the RRF paper's value, and what most production systems that cite it
use) dampens the difference between e.g. rank 1 and rank 2 — without it, a
rank-1-in-one-list item could dominate a rank-2-in-both-lists item that's
arguably more robustly relevant.
"""

from uuid import UUID


def reciprocal_rank_fusion(
    ranked_lists: list[list[UUID]], k: int = 60
) -> list[tuple[UUID, float]]:
    """Takes any number of ranked chunk-ID lists (each already sorted best
    first) and returns a single fused ranking as (chunk_id, rrf_score)
    pairs, sorted best first. A chunk_id missing from a list simply
    contributes nothing from that list — no penalty beyond "no bonus"."""
    scores: dict[UUID, float] = {}
    for ranked_list in ranked_lists:
        for position, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
