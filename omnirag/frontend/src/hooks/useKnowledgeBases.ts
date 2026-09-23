import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import * as api from '@/lib/api'

export const knowledgeBaseKeys = {
  all: ['knowledge-bases'] as const,
  detail: (id: string) => ['knowledge-bases', id] as const,
}

export function useKnowledgeBases() {
  return useQuery({
    queryKey: knowledgeBaseKeys.all,
    queryFn: api.listKnowledgeBases,
  })
}

export function useKnowledgeBase(id: string | undefined) {
  return useQuery({
    queryKey: knowledgeBaseKeys.detail(id ?? ''),
    queryFn: () => api.getKnowledgeBase(id as string),
    enabled: Boolean(id),
  })
}

export function useCreateKnowledgeBase() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ name, description }: { name: string; description?: string }) =>
      api.createKnowledgeBase(name, description),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: knowledgeBaseKeys.all })
    },
  })
}

export function useDeleteKnowledgeBase() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.deleteKnowledgeBase(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: knowledgeBaseKeys.all })
    },
  })
}
