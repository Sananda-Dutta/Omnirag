import type { ReactNode } from 'react'
import { Navigate, useLocation } from 'react-router-dom'

import { Spinner } from '@/components/common/Spinner'
import { useAuth } from '@/context/AuthContext'

export function ProtectedRoute({ children }: { children: ReactNode }) {
  const { user, isLoading } = useAuth()
  const location = useLocation()

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-ink-950">
        <Spinner label="Loading OmniRAG..." />
      </div>
    )
  }

  if (!user) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />
  }

  return <>{children}</>
}
