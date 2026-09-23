import { useState } from 'react'
import { Link } from 'react-router-dom'

import { Button } from '@/components/common/Button'
import { EmptyState } from '@/components/common/EmptyState'
import { ErrorBanner } from '@/components/common/ErrorBanner'
import { Spinner } from '@/components/common/Spinner'
import { TextField } from '@/components/common/TextField'
import { useCreateKnowledgeBase, useKnowledgeBases } from '@/hooks/useKnowledgeBases'
import { extractErrorMessage } from '@/lib/api'

function CreateKnowledgeBaseForm() {
  const [name, setName] = useState('')
  const [isOpen, setIsOpen] = useState(false)
  const createMutation = useCreateKnowledgeBase()

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault()
    if (!name.trim()) return
    createMutation.mutate(
      { name: name.trim() },
      { onSuccess: () => { setName(''); setIsOpen(false) } },
    )
  }

  if (!isOpen) {
    return (
      <Button type="button" onClick={() => setIsOpen(true)}>
        New knowledge base
      </Button>
    )
  }

  return (
    <form onSubmit={handleSubmit} className="flex items-end gap-2">
      <TextField
        label="Name"
        autoFocus
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="e.g. Product Docs"
      />
      <Button type="submit" isLoading={createMutation.isPending}>
        Create
      </Button>
      <Button type="button" variant="ghost" onClick={() => setIsOpen(false)}>
        Cancel
      </Button>
      {createMutation.isError && <ErrorBanner message={extractErrorMessage(createMutation.error)} />}
    </form>
  )
}

export function KnowledgeBaseList() {
  const { data: knowledgeBases, isLoading, isError } = useKnowledgeBases()

  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center justify-between">
        <h1 className="font-display text-2xl text-paper-100">Knowledge bases</h1>
        <CreateKnowledgeBaseForm />
      </div>

      {isLoading && <Spinner label="Loading knowledge bases..." />}
      {isError && <ErrorBanner message="Couldn't load your knowledge bases." />}

      {knowledgeBases && knowledgeBases.length === 0 && (
        <EmptyState
          title="No knowledge bases yet"
          description="Create one to start uploading documents and asking questions grounded in them."
        />
      )}

      {knowledgeBases && knowledgeBases.length > 0 && (
        <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {knowledgeBases.map((kb) => (
            <li key={kb.id}>
              <Link
                to={`/kb/${kb.id}`}
                className="block h-full rounded-sm border border-ink-700 bg-ink-900/60 p-4 transition-colors hover:border-gold-500/50"
              >
                <p className="font-display text-lg text-paper-100">{kb.name}</p>
                {kb.description && (
                  <p className="mt-1 line-clamp-2 text-sm text-paper-muted">{kb.description}</p>
                )}
                <p className="mt-3 font-mono text-[11px] text-paper-faint">
                  created {new Date(kb.created_at).toLocaleDateString()}
                </p>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
