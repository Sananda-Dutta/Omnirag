import { useMutation } from '@tanstack/react-query'

import * as api from '@/lib/api'

export function useSendChatMessage() {
  return useMutation({
    mutationFn: ({ question, knowledgeBaseId }: { question: string; knowledgeBaseId?: string }) =>
      api.chat(question, knowledgeBaseId),
  })
}
