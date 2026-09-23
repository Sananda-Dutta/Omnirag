import { AppShell } from '@/components/layout/AppShell'
import { KnowledgeBaseList } from '@/components/knowledge-base/KnowledgeBaseList'

export function DashboardPage() {
  return (
    <AppShell>
      <KnowledgeBaseList />
    </AppShell>
  )
}
