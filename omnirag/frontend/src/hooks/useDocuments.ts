import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import * as api from '@/lib/api'
import type { DocumentSummary } from '@/lib/types'

export const documentKeys = {
  forKnowledgeBase: (kbId: string) => ['documents', kbId] as const,
  detail: (id: string) => ['documents', 'detail', id] as const,
  chunks: (id: string) => ['documents', 'chunks', id] as const,
}

const ACTIVE_STATUSES = new Set<DocumentSummary['status']>(['pending', 'processing'])

export function useDocuments(knowledgeBaseId: string) {
  return useQuery({
    queryKey: documentKeys.forKnowledgeBase(knowledgeBaseId),
    queryFn: () => api.listDocuments(knowledgeBaseId),
    // Ingestion runs on a Celery worker, not synchronously with the
    // upload request — poll while anything is still pending/processing so
    // a freshly-uploaded document's status visibly updates without the
    // user needing to refresh. Stops polling once nothing is in flight.
    refetchInterval: (query) => {
      const docs = query.state.data
      return docs?.some((d) => ACTIVE_STATUSES.has(d.status)) ? 2000 : false
    },
  })
}

export function useDocument(documentId: string | undefined) {
  return useQuery({
    queryKey: documentKeys.detail(documentId ?? ''),
    queryFn: () => api.getDocument(documentId as string),
    enabled: Boolean(documentId),
  })
}

export function useChunks(documentId: string | undefined) {
  return useQuery({
    queryKey: documentKeys.chunks(documentId ?? ''),
    queryFn: () => api.listChunks(documentId as string),
    enabled: Boolean(documentId),
  })
}

export function useUploadDocument(knowledgeBaseId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (file: File) => api.uploadDocument(knowledgeBaseId, file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: documentKeys.forKnowledgeBase(knowledgeBaseId) })
    },
  })
}

export function useIngestUrl(knowledgeBaseId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (url: string) => api.ingestUrl(knowledgeBaseId, url),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: documentKeys.forKnowledgeBase(knowledgeBaseId) })
    },
  })
}

export function useDeleteDocument(knowledgeBaseId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (documentId: string) => api.deleteDocument(documentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: documentKeys.forKnowledgeBase(knowledgeBaseId) })
    },
  })
}
