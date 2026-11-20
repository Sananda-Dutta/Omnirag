import { Modal } from '@/components/common/Modal'
import { Spinner } from '@/components/common/Spinner'
import { useChunks } from '@/hooks/useDocuments'

export function ChunkViewerModal({
  documentId,
  filename,
  highlightChunkId,
  onClose,
}: {
  documentId: string
  filename: string
  /** Pre-scrolls to / highlights the chunk a citation click came from. */
  highlightChunkId?: string
  onClose: () => void
}) {
  const { data: chunks, isLoading, isError } = useChunks(documentId)

  return (
    <Modal title={filename} onClose={onClose}>
      {isLoading && <Spinner label="Loading chunks..." />}
      {isError && <p className="text-sm text-rust-400">Couldn't load this document's chunks.</p>}
      {chunks && chunks.length === 0 && (
        <p className="text-sm text-paper-muted">This document has no chunks yet.</p>
      )}
      <ul className="flex flex-col gap-3">
        {chunks?.map((chunk) => (
          <li
            key={chunk.id}
            className={`rounded-sm border p-3 text-sm leading-relaxed ${
              chunk.id === highlightChunkId
                ? 'border-gold-500/60 bg-gold-500/5'
                : 'border-ink-700 bg-ink-850'
            }`}
          >
            <div className="mb-1.5 flex items-center gap-2 font-mono text-xs text-paper-faint">
              <span>chunk {chunk.chunk_index}</span>
              {chunk.page_number !== null && <span>· page {chunk.page_number}</span>}
              <span>· {chunk.char_count} chars</span>
            </div>
            <p className="whitespace-pre-wrap text-paper-100">{chunk.text}</p>
          </li>
        ))}
      </ul>
    </Modal>
  )
}
