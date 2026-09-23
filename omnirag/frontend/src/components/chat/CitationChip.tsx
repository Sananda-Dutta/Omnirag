import type { Citation } from '@/lib/types'

export function CitationChip({
  citation,
  index,
  onClick,
}: {
  citation: Citation
  index: number
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={citation.text_snippet}
      className="inline-flex items-center gap-1.5 rounded-full border border-ink-600 bg-ink-850
        px-2.5 py-1 font-mono text-xs text-paper-muted transition-colors hover:border-gold-500/60 hover:text-gold-400"
    >
      <span className="text-gold-500">[{index}]</span>
      <span className="max-w-40 truncate">{citation.document_filename}</span>
      {citation.page_number !== null && <span className="text-paper-faint">p.{citation.page_number}</span>}
    </button>
  )
}
