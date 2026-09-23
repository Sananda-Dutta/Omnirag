import uuid

from pydantic import BaseModel, Field


class EvalQuestion(BaseModel):
    """One row of an evaluation dataset. Only `question` is required —
    everything else narrows which metrics can be computed for this row
    (see EvalQuestionResult): no `expected_document_filenames` means no
    retrieval metrics for this row, no `expected_answer` means no
    answer_correctness for this row. A real eval set commonly mixes rows
    that have a labeled answer with rows that only check retrieval, or
    vice versa — this isn't an error case to special-case, it's normal."""

    question: str = Field(min_length=1, max_length=2000)
    expected_answer: str | None = Field(
        default=None, description="Known-correct reference answer, if you have one. Enables answer_correctness."
    )
    expected_document_filenames: list[str] | None = Field(
        default=None,
        description="Filenames of documents that should be among the retrieved "
        "sources for this question. Enables precision@k/recall@k/MRR/hit-rate.",
    )


class EvalRunRequest(BaseModel):
    knowledge_base_id: uuid.UUID | None = Field(
        default=None, description="Restrict retrieval to one knowledge base, same as ChatRequest. Omit to search all of the user's."
    )
    questions: list[EvalQuestion] = Field(min_length=1, max_length=200)
    top_k: int | None = Field(
        default=None, ge=1, le=50, description="Overrides RAG_TOP_K for this eval run only."
    )


class RetrievalScores(BaseModel):
    precision_at_k: float
    recall_at_k: float
    mrr: float
    hit_rate: float


class GenerationScoresOut(BaseModel):
    faithfulness: float | None
    answer_relevancy: float | None
    answer_correctness: float | None


class EvalQuestionResult(BaseModel):
    question: str
    answer: str
    model_used: str
    context_found: bool
    retrieved_document_filenames: list[str]
    retrieval: RetrievalScores | None = Field(
        default=None, description="Null when the question didn't supply expected_document_filenames."
    )
    generation: GenerationScoresOut


class EvalSummary(BaseModel):
    question_count: int
    context_found_rate: float = Field(description="Fraction of questions where any context was retrieved at all.")
    avg_precision_at_k: float | None
    avg_recall_at_k: float | None
    avg_mrr: float | None
    avg_hit_rate: float | None
    avg_faithfulness: float | None
    avg_answer_relevancy: float | None
    avg_answer_correctness: float | None


class EvalReport(BaseModel):
    eval_provider: str
    top_k: int
    results: list[EvalQuestionResult]
    summary: EvalSummary
