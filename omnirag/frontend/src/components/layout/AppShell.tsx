import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'

import { useAuth } from '@/context/AuthContext'

export function AppShell({ children }: { children: ReactNode }) {
  const { user, logout } = useAuth()

  return (
    <div className="min-h-screen bg-ink-950">
      <header className="border-b border-ink-700">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <Link to="/" className="flex items-baseline gap-2">
            <span className="font-display text-lg text-paper-100">OmniRAG</span>
            <span className="font-mono text-[11px] text-paper-faint">multi-modal knowledge assistant</span>
          </Link>
          {user && (
            <div className="flex items-center gap-4 text-sm">
              <span className="text-paper-muted">{user.email}</span>
              <button type="button" onClick={logout} className="text-paper-faint hover:text-paper-100">
                Log out
              </button>
            </div>
          )}
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
    </div>
  )
}
