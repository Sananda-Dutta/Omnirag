import uuid

from app.retrieval.fusion import reciprocal_rank_fusion


def test_item_in_both_lists_outranks_item_in_only_one():
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    # a: rank 1 in dense only. b: rank 2 in both. c: rank 3 in keyword only.
    dense = [a, b]
    keyword = [b, c]

    fused = reciprocal_rank_fusion([dense, keyword])
    fused_ids = [chunk_id for chunk_id, _ in fused]

    # b appears in both lists (even at rank 2 in each), so it should rank
    # above a, which only appears in one list at rank 1.
    assert fused_ids.index(b) < fused_ids.index(a)


def test_single_list_preserves_original_order():
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    fused = reciprocal_rank_fusion([[a, b, c]])
    assert [chunk_id for chunk_id, _ in fused] == [a, b, c]


def test_empty_lists_produce_empty_result():
    assert reciprocal_rank_fusion([[], []]) == []


def test_disjoint_lists_are_still_merged():
    a, b = uuid.uuid4(), uuid.uuid4()
    fused = reciprocal_rank_fusion([[a], [b]])
    assert {chunk_id for chunk_id, _ in fused} == {a, b}


def test_rank_1_scores_higher_than_rank_2_within_same_list():
    a, b = uuid.uuid4(), uuid.uuid4()
    fused = dict(reciprocal_rank_fusion([[a, b]]))
    assert fused[a] > fused[b]


def test_smaller_k_amplifies_rank_differences():
    a, b = uuid.uuid4(), uuid.uuid4()
    fused_k1 = dict(reciprocal_rank_fusion([[a, b]], k=1))
    fused_k60 = dict(reciprocal_rank_fusion([[a, b]], k=60))

    ratio_k1 = fused_k1[a] / fused_k1[b]
    ratio_k60 = fused_k60[a] / fused_k60[b]
    assert ratio_k1 > ratio_k60
