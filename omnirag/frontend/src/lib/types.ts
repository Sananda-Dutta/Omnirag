// Mirrors backend/app/schemas/*.py. Kept as one file since the backend
// keeps each schema module small and this project has no codegen step
// (see api.ts's docstring for why) — if the backend schemas grow
// significantly, split this the same way the backend does.

export interface User {
  id: string
  email: string
  is_active: boolean
  created_at: string
}

export interface KnowledgeBase {
  id: string
  name: string
  description: string | null
  created_at: string
  updated_at: string
}

export type DocumentStatus = 'pending' | 'processing' | 'completed' | 'failed'

export interface DocumentSummary {
  id: string
  knowledge_base_id: string
  filename: string
  content_type: string
  file_size_bytes: number | null
  source_url: string | null
  status: DocumentStatus
  error_message: string | null
  page_count: number | null
  char_count: number | null
  created_at: string
  updated_at: string
}

export interface DocumentDetail extends DocumentSummary {
  extracted_text: string | null
}

export interface Chunk {
  id: string
  chunk_index: number
  text: string
  char_count: number
  page_number: number | null
  embedding_model: string
  embedding_dimension: number
  created_at: string
}

export interface SearchResult {
  chunk_id: string
  score: number
  text: string
  chunk_index: number
  page_number: number | null
  document_id: string
  document_filename: string
}

export interface Citation {
  chunk_id: string
  document_id: string
  document_filename: string
  chunk_index: number
  page_number: number | null
  score: number
  text_snippet: string
}

export interface ChatResponse {
  answer: string
  citations: Citation[]
  model_used: string
  context_found: boolean
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  modelUsed?: string
  contextFound?: boolean
  pending?: boolean
  error?: string
}

export interface EvalQuestionInput {
  question: string
  expected_answer?: string
  expected_document_filenames?: string[]
}

export interface RetrievalScores {
  precision_at_k: number
  recall_at_k: number
  mrr: number
  hit_rate: number
}

export interface GenerationScores {
  faithfulness: number | null
  answer_relevancy: number | null
  answer_correctness: number | null
}

export interface EvalQuestionResult {
  question: string
  answer: string
  model_used: string
  context_found: boolean
  retrieved_document_filenames: string[]
  retrieval: RetrievalScores | null
  generation: GenerationScores
}

export interface EvalSummary {
  question_count: number
  context_found_rate: number
  avg_precision_at_k: number | null
  avg_recall_at_k: number | null
  avg_mrr: number | null
  avg_hit_rate: number | null
  avg_faithfulness: number | null
  avg_answer_relevancy: number | null
  avg_answer_correctness: number | null
}

export interface EvalReport {
  eval_provider: string
  top_k: number
  results: EvalQuestionResult[]
  summary: EvalSummary
}

export interface ApiErrorBody {
  detail?: string | { msg: string; loc: (string | number)[] }[]
}
