import { useState } from 'react'

import { ChunkViewerModal } from '@/components/documents/ChunkViewerModal'
import { DocumentStatusBadge } from '@/components/documents/DocumentStatusBadge'
import { Button } from '@/components/common/Button'
import { EmptyState } from '@/components/common/EmptyState'
import { Spinner } from '@/components/common/Spinner'
import { useDeleteDocument, useDocuments } from '@/hooks/useDocuments'
import type { DocumentSummary } from '@/lib/types'

function formatBytes(bytes: number | null): string {
  if (bytes === null) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function DocumentList({ knowledgeBaseId }: { knowledgeBaseId: string }) {
  const { data: documents, isLoading } = useDocuments(knowledgeBaseId)
  const deleteMutation = useDeleteDocument(knowledgeBaseId)
  const [openDocument, setOpenDocument] = useState<DocumentSummary | null>(null)

  if (isLoading) return <Spinner label="Loading documents..." />

  if (!documents || documents.length === 0) {
    return (
      <EmptyState
        title="No documents yet"
        description="Upload a file or add a URL above to start building this knowledge base."
      />
    )
  }

  return (
    <>
      <ul className="flex flex-col divide-y divide-ink-700 rounded-sm border border-ink-700">
        {documents.map((doc) => (
          <li key={doc.id} className="flex items-center justify-between gap-4 px-4 py-3">
            <button
              type="button"
              onClick={() => doc.status === 'completed' && setOpenDocument(doc)}
              disabled={doc.status !== 'completed'}
              className="min-w-0 flex-1 text-left disabled:cursor-default"
            >
              <p className="truncate text-sm text-paper-100">{doc.filename}</p>
              <p className="mt-0.5 truncate font-mono text-xs text-paper-faint">
                {doc.source_url ? doc.source_url : doc.content_type}
                {' · '}
                {formatBytes(doc.file_size_bytes)}
                {doc.char_count !== null && ` · ${doc.char_count.toLocaleString()} chars`}
                {doc.page_count !== null && ` · ${doc.page_count} pages`}
              </p>
            </button>
            <DocumentStatusBadge status={doc.status} errorMessage={doc.error_message} />
            <Button
              type="button"
              variant="ghost"
              onClick={() => deleteMutation.mutate(doc.id)}
              isLoading={deleteMutation.isPending && deleteMutation.variables === doc.id}
            >
              Remove
            </Button>
          </li>
        ))}
      </ul>

      {openDocument && (
        <ChunkViewerModal
          documentId={openDocument.id}
          filename={openDocument.filename}
          onClose={() => setOpenDocument(null)}
        />
      )}
    </>
  )
}
