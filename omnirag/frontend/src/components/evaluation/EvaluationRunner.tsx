import { useState } from 'react'

import { EvalResultsTable } from '@/components/evaluation/EvalResultsTable'
import { Button } from '@/components/common/Button'
import { ErrorBanner } from '@/components/common/ErrorBanner'
import { TextField } from '@/components/common/TextField'
import { useRunEvaluation } from '@/hooks/useEvaluation'
import { extractErrorMessage } from '@/lib/api'
import type { EvalQuestionInput } from '@/lib/types'

interface DraftQuestion {
  key: string
  question: string
  expectedAnswer: string
  expectedFilenames: string
}

function emptyDraft(): DraftQuestion {
  return { key: crypto.randomUUID(), question: '', expectedAnswer: '', expectedFilenames: '' }
}

export function EvaluationRunner({ knowledgeBaseId }: { knowledgeBaseId: string }) {
  const [drafts, setDrafts] = useState<DraftQuestion[]>([emptyDraft()])
  const runEvaluation = useRunEvaluation()

  const updateDraft = (key: string, patch: Partial<DraftQuestion>) => {
    setDrafts((prev) => prev.map((d) => (d.key === key ? { ...d, ...patch } : d)))
  }

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault()
    const questions: EvalQuestionInput[] = drafts
      .filter((d) => d.question.trim())
      .map((d) => ({
        question: d.question.trim(),
        expected_answer: d.expectedAnswer.trim() || undefined,
        expected_document_filenames: d.expectedFilenames.trim()
          ? d.expectedFilenames.split(',').map((f) => f.trim()).filter(Boolean)
          : undefined,
      }))
    if (questions.length === 0) return
    runEvaluation.mutate({ questions, knowledgeBaseId })
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="rounded-sm border border-ink-700 bg-ink-900/60 p-4">
        <p className="mb-1 font-display text-base text-paper-100">Build a quick eval set</p>
        <p className="mb-4 text-sm text-paper-muted">
          Add a few questions this knowledge base should be able to answer. A reference answer and
          the filenames you expect to be cited are both optional — retrieval and generation are only
          scored for what you provide.
        </p>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          {drafts.map((draft, i) => (
            <div key={draft.key} className="flex flex-col gap-2 border-l-2 border-ink-700 pl-3">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs text-paper-faint">question {i + 1}</span>
                {drafts.length > 1 && (
                  <button
                    type="button"
                    onClick={() => setDrafts((prev) => prev.filter((d) => d.key !== draft.key))}
                    className="text-xs text-paper-faint hover:text-rust-400"
                  >
                    remove
                  </button>
                )}
              </div>
              <TextField
                placeholder="What does...?"
                value={draft.question}
                onChange={(e) => updateDraft(draft.key, { question: e.target.value })}
              />
              <TextField
                placeholder="Reference answer (optional)"
                value={draft.expectedAnswer}
                onChange={(e) => updateDraft(draft.key, { expectedAnswer: e.target.value })}
              />
              <TextField
                placeholder="Expected source filenames, comma-separated (optional)"
                value={draft.expectedFilenames}
                onChange={(e) => updateDraft(draft.key, { expectedFilenames: e.target.value })}
              />
            </div>
          ))}

          <div className="flex items-center gap-2">
            <Button type="button" variant="secondary" onClick={() => setDrafts((prev) => [...prev, emptyDraft()])}>
              + Add question
            </Button>
            <Button type="submit" isLoading={runEvaluation.isPending}>
              Run evaluation
            </Button>
          </div>
        </form>

        {runEvaluation.isError && (
          <div className="mt-3">
            <ErrorBanner message={extractErrorMessage(runEvaluation.error)} />
          </div>
        )}
      </div>

      {runEvaluation.data && <EvalResultsTable report={runEvaluation.data} />}
    </div>
  )
}
