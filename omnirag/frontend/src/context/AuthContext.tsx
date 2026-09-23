import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'

import * as api from '@/lib/api'
import { getStoredToken, setStoredToken } from '@/lib/api'
import type { User } from '@/lib/types'

interface AuthContextValue {
  user: User | null
  /** True while we're checking a stored token against /auth/me on first
   * load — distinct from "not logged in", so a page can show a loading
   * state instead of flashing the login screen for someone who's actually
   * already signed in. */
  isLoading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const token = getStoredToken()
    if (!token) {
      setIsLoading(false)
      return
    }
    api
      .fetchCurrentUser()
      .then(setUser)
      .catch(() => {
        // Stored token is expired/invalid — clear it rather than leave the
        // app thinking it has a session that every subsequent call will 401 on.
        setStoredToken(null)
      })
      .finally(() => setIsLoading(false))
  }, [])

  const login = useCallback(async (email: string, password: string) => {
    const token = await api.login(email, password)
    setStoredToken(token)
    const currentUser = await api.fetchCurrentUser()
    setUser(currentUser)
  }, [])

  const register = useCallback(async (email: string, password: string) => {
    await api.register(email, password)
    // Registration doesn't itself return a token — log in right after so
    // "create an account" reads as one action to the person, not two.
    await login(email, password)
  }, [login])

  const logout = useCallback(() => {
    setStoredToken(null)
    setUser(null)
  }, [])

  return (
    <AuthContext.Provider value={{ user, isLoading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within an AuthProvider')
  return ctx
}
