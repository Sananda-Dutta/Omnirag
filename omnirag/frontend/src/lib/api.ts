import axios, { AxiosError } from 'axios'

import type {
  ApiErrorBody,
  ChatResponse,
  Chunk,
  DocumentDetail,
  DocumentSummary,
  EvalQuestionInput,
  EvalReport,
  KnowledgeBase,
  SearchResult,
  User,
} from '@/lib/types'

const TOKEN_STORAGE_KEY = 'omnirag_access_token'

export function getStoredToken(): string | null {
  return localStorage.getItem(TOKEN_STORAGE_KEY)
}

export function setStoredToken(token: string | null): void {
  if (token) {
    localStorage.setItem(TOKEN_STORAGE_KEY, token)
  } else {
    localStorage.removeItem(TOKEN_STORAGE_KEY)
  }
}

const client = axios.create({ baseURL: '/api/v1' })

client.interceptors.request.use((config) => {
  const token = getStoredToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

/** Every backend error is FastAPI's standard {"detail": "..."} or, for a
 * 422 validation error, {"detail": [{"msg": ..., "loc": [...]}]} — this
 * turns either shape into one readable string instead of making every
 * call site branch on response shape. */
export function extractErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<ApiErrorBody>
    const detail = axiosError.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail) && detail.length > 0) {
      return detail.map((d) => d.msg).join('; ')
    }
    if (axiosError.response?.status === 401) return 'Your session has expired. Please log in again.'
    if (axiosError.code === 'ERR_NETWORK') return "Can't reach the server. Is the backend running?"
  }
  return 'Something went wrong. Please try again.'
}

// --- Auth --------------------------------------------------------------

export async function register(email: string, password: string): Promise<User> {
  const { data } = await client.post<User>('/auth/register', { email, password })
  return data
}

export async function login(email: string, password: string): Promise<string> {
  // /auth/login expects OAuth2's form-encoded username/password, not JSON
  // — see backend/app/api/auth.py's docstring for why.
  const form = new URLSearchParams()
  form.set('username', email)
  form.set('password', password)
  const { data } = await client.post<{ access_token: string }>('/auth/login', form, {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  })
  return data.access_token
}

export async function fetchCurrentUser(): Promise<User> {
  const { data } = await client.get<User>('/auth/me')
  return data
}

// --- Knowledge bases -----------------------------------------------------

export async function listKnowledgeBases(): Promise<KnowledgeBase[]> {
  const { data } = await client.get<KnowledgeBase[]>('/knowledge-bases')
  return data
}

export async function createKnowledgeBase(name: string, description?: string): Promise<KnowledgeBase> {
  const { data } = await client.post<KnowledgeBase>('/knowledge-bases', { name, description })
  return data
}

export async function getKnowledgeBase(id: string): Promise<KnowledgeBase> {
  const { data } = await client.get<KnowledgeBase>(`/knowledge-bases/${id}`)
  return data
}

export async function deleteKnowledgeBase(id: string): Promise<void> {
  await client.delete(`/knowledge-bases/${id}`)
}

// --- Documents -------------------------------------------------------------

export async function listDocuments(knowledgeBaseId: string): Promise<DocumentSummary[]> {
  const { data } = await client.get<DocumentSummary[]>(`/knowledge-bases/${knowledgeBaseId}/documents`)
  return data
}

export async function uploadDocument(knowledgeBaseId: string, file: File): Promise<DocumentSummary> {
  const form = new FormData()
  form.set('file', file)
  const { data } = await client.post<DocumentSummary>(
    `/knowledge-bases/${knowledgeBaseId}/documents`,
    form,
    { headers: { 'Content-Type': 'multipart/form-data' } },
  )
  return data
}

export async function ingestUrl(knowledgeBaseId: string, url: string): Promise<DocumentSummary> {
  const { data } = await client.post<DocumentSummary>(
    `/knowledge-bases/${knowledgeBaseId}/documents/url`,
    { url },
  )
  return data
}

export async function getDocument(documentId: string): Promise<DocumentDetail> {
  const { data } = await client.get<DocumentDetail>(`/documents/${documentId}`)
  return data
}

export async function deleteDocument(documentId: string): Promise<void> {
  await client.delete(`/documents/${documentId}`)
}

export async function listChunks(documentId: string): Promise<Chunk[]> {
  const { data } = await client.get<Chunk[]>(`/documents/${documentId}/chunks`)
  return data
}

export async function getChunk(documentId: string, chunkId: string): Promise<Chunk> {
  const { data } = await client.get<Chunk>(`/documents/${documentId}/chunks/${chunkId}`)
  return data
}

// --- Search / chat -----------------------------------------------------

export async function search(
  query: string,
  knowledgeBaseId?: string,
  topK?: number,
): Promise<SearchResult[]> {
  const { data } = await client.post<SearchResult[]>('/search', {
    query,
    knowledge_base_id: knowledgeBaseId,
    top_k: topK,
  })
  return data
}

export async function chat(question: string, knowledgeBaseId?: string): Promise<ChatResponse> {
  const { data } = await client.post<ChatResponse>('/chat', {
    question,
    knowledge_base_id: knowledgeBaseId,
  })
  return data
}

// --- Evaluation (Phase 15) --------------------------------------------

export async function runEvaluation(
  questions: EvalQuestionInput[],
  knowledgeBaseId?: string,
  topK?: number,
): Promise<EvalReport> {
  const { data } = await client.post<EvalReport>('/evaluation/run', {
    questions,
    knowledge_base_id: knowledgeBaseId,
    top_k: topK,
  })
  return data
}

export default client
