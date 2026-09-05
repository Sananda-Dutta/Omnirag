import uuid

from app.retrieval.reranker import LocalLexicalReranker, RerankCandidate


def test_more_matching_terms_ranks_higher():
    reranker = LocalLexicalReranker()
    strong = RerankCandidate(chunk_id=uuid.uuid4(), text="Gradient descent minimizes the loss function.")
    weak = RerankCandidate(chunk_id=uuid.uuid4(), text="Gradient descent is one topic among many in this course.")
    unrelated = RerankCandidate(chunk_id=uuid.uuid4(), text="The bakery opens at seven in the morning.")

    result = reranker.rerank("gradient descent loss function", [unrelated, weak, strong])
    ordered_ids = [chunk_id for chunk_id, _ in result]

    assert ordered_ids[0] == strong.chunk_id
    assert ordered_ids[-1] == unrelated.chunk_id


def test_rare_shared_term_weighted_higher_than_common_one():
    reranker = LocalLexicalReranker()
    # "transformer" appears in only one candidate (rare -> discriminating);
    # "model" appears in both (common -> less discriminating).
    rare_match = RerankCandidate(chunk_id=uuid.uuid4(), text="The transformer model uses self-attention.")
    common_only = RerankCandidate(chunk_id=uuid.uuid4(), text="This model is a simple linear model.")

    result = reranker.rerank("transformer model", [common_only, rare_match])
    ordered_ids = [chunk_id for chunk_id, _ in result]

    assert ordered_ids[0] == rare_match.chunk_id


def test_empty_candidates_returns_empty():
    assert LocalLexicalReranker().rerank("anything", []) == []


def test_query_with_no_real_words_returns_original_order_with_zero_scores():
    reranker = LocalLexicalReranker()
    a = RerankCandidate(chunk_id=uuid.uuid4(), text="Some content here.")
    b = RerankCandidate(chunk_id=uuid.uuid4(), text="Other content here.")
    result = reranker.rerank("???", [a, b])
    assert [chunk_id for chunk_id, _ in result] == [a.chunk_id, b.chunk_id]
    assert all(score == 0.0 for _, score in result)


def test_candidate_with_no_overlap_scores_zero():
    reranker = LocalLexicalReranker()
    candidate = RerankCandidate(chunk_id=uuid.uuid4(), text="completely unrelated content")
    result = reranker.rerank("gradient descent", [candidate])
    assert result[0][1] == 0.0
