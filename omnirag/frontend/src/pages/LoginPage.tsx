import { useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'

import { Button } from '@/components/common/Button'
import { ErrorBanner } from '@/components/common/ErrorBanner'
import { TextField } from '@/components/common/TextField'
import { useAuth } from '@/context/AuthContext'
import { extractErrorMessage } from '@/lib/api'

export function LoginPage() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const redirectTo = (location.state as { from?: string } | null)?.from ?? '/'

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    setError(null)
    setIsSubmitting(true)
    try {
      await login(email, password)
      navigate(redirectTo, { replace: true })
    } catch (err) {
      setError(extractErrorMessage(err))
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-ink-950 px-4">
      <div className="w-full max-w-sm">
        <p className="mb-1 text-center font-display text-2xl text-paper-100">OmniRAG</p>
        <p className="mb-8 text-center text-sm text-paper-muted">Sign in to your knowledge bases</p>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <TextField
            label="Email"
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <TextField
            label="Password"
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          {error && <ErrorBanner message={error} />}
          <Button type="submit" isLoading={isSubmitting} className="mt-2 w-full">
            Sign in
          </Button>
        </form>

        <p className="mt-6 text-center text-sm text-paper-muted">
          New here?{' '}
          <Link to="/register" className="text-gold-400 hover:underline">
            Create an account
          </Link>
        </p>
      </div>
    </div>
  )
}
