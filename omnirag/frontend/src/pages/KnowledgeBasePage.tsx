import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import { AppShell } from '@/components/layout/AppShell'
import { UploadPanel } from '@/components/documents/UploadPanel'
import { DocumentList } from '@/components/documents/DocumentList'
import { ChatPanel } from '@/components/chat/ChatPanel'
import { EvaluationRunner } from '@/components/evaluation/EvaluationRunner'
import { Button } from '@/components/common/Button'
import { Spinner } from '@/components/common/Spinner'
import { useDeleteKnowledgeBase, useKnowledgeBase } from '@/hooks/useKnowledgeBases'

type Tab = 'documents' | 'chat' | 'evaluate'

const TABS: { id: Tab; label: string }[] = [
  { id: 'documents', label: 'Documents' },
  { id: 'chat', label: 'Chat' },
  { id: 'evaluate', label: 'Evaluate' },
]

export function KnowledgeBasePage() {
  const { knowledgeBaseId } = useParams<{ knowledgeBaseId: string }>()
  const navigate = useNavigate()
  const [tab, setTab] = useState<Tab>('documents')
  const { data: knowledgeBase, isLoading } = useKnowledgeBase(knowledgeBaseId)
  const deleteMutation = useDeleteKnowledgeBase()

  if (!knowledgeBaseId) return null

  const handleDelete = () => {
    if (!window.confirm('Delete this knowledge base and everything in it? This can’t be undone.')) return
    deleteMutation.mutate(knowledgeBaseId, { onSuccess: () => navigate('/') })
  }

  return (
    <AppShell>
      <Link to="/" className="mb-4 inline-block text-sm text-paper-faint hover:text-paper-100">
        ← All knowledge bases
      </Link>

      {isLoading ? (
        <Spinner label="Loading..." />
      ) : (
        <>
          <div className="mb-6 flex items-start justify-between gap-4">
            <div>
              <h1 className="font-display text-2xl text-paper-100">{knowledgeBase?.name}</h1>
              {knowledgeBase?.description && (
                <p className="mt-1 text-sm text-paper-muted">{knowledgeBase.description}</p>
              )}
            </div>
            <Button variant="danger" onClick={handleDelete} isLoading={deleteMutation.isPending}>
              Delete
            </Button>
          </div>

          <div className="mb-6 flex gap-1 border-b border-ink-700">
            {TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => setTab(t.id)}
                className={`px-3 py-2 text-sm transition-colors ${
                  tab === t.id
                    ? 'border-b-2 border-gold-500 text-paper-100'
                    : 'text-paper-muted hover:text-paper-100'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          {tab === 'documents' && (
            <div className="flex flex-col gap-5">
              <UploadPanel knowledgeBaseId={knowledgeBaseId} />
              <DocumentList knowledgeBaseId={knowledgeBaseId} />
            </div>
          )}
          {tab === 'chat' && <ChatPanel knowledgeBaseId={knowledgeBaseId} />}
          {tab === 'evaluate' && <EvaluationRunner knowledgeBaseId={knowledgeBaseId} />}
        </>
      )}
    </AppShell>
  )
}
