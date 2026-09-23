import type { DocumentStatus } from '@/lib/types'

const STATUS_META: Record<DocumentStatus, { label: string; dot: string; text: string }> = {
  pending: { label: 'Queued', dot: 'bg-paper-faint', text: 'text-paper-muted' },
  processing: { label: 'Processing', dot: 'bg-gold-500 animate-pulse', text: 'text-gold-400' },
  completed: { label: 'Ready', dot: 'bg-moss-500', text: 'text-moss-400' },
  failed: { label: 'Failed', dot: 'bg-rust-500', text: 'text-rust-400' },
}

export function DocumentStatusBadge({
  status,
  errorMessage,
}: {
  status: DocumentStatus
  errorMessage?: string | null
}) {
  const meta = STATUS_META[status]
  return (
    <span
      className={`inline-flex items-center gap-1.5 text-xs font-mono ${meta.text}`}
      title={status === 'failed' ? (errorMessage ?? undefined) : undefined}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${meta.dot}`} aria-hidden="true" />
      {meta.label}
    </span>
  )
}
