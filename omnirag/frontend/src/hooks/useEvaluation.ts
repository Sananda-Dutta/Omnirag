import { useMutation } from '@tanstack/react-query'

import * as api from '@/lib/api'
import type { EvalQuestionInput } from '@/lib/types'

export function useRunEvaluation() {
  return useMutation({
    mutationFn: ({
      questions,
      knowledgeBaseId,
      topK,
    }: {
      questions: EvalQuestionInput[]
      knowledgeBaseId?: string
      topK?: number
    }) => api.runEvaluation(questions, knowledgeBaseId, topK),
  })
}
