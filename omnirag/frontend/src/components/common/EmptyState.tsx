import type { ReactNode } from 'react'

interface EmptyStateProps {
  title: string
  description?: string
  action?: ReactNode
}

/** An empty screen is an invitation to act, not just an absence notice —
 * every call site should pair this with something the person can do next. */
export function EmptyState({ title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-sm border border-dashed border-ink-600 px-6 py-14 text-center">
      <p className="font-display text-lg text-paper-100">{title}</p>
      {description && <p className="max-w-sm text-sm text-paper-muted">{description}</p>}
      {action}
    </div>
  )
}
