import { useEffect, useRef, useState } from 'react'

import { ChunkViewerModal } from '@/components/documents/ChunkViewerModal'
import { MessageBubble } from '@/components/chat/MessageBubble'
import { Button } from '@/components/common/Button'
import { EmptyState } from '@/components/common/EmptyState'
import { useSendChatMessage } from '@/hooks/useChat'
import { extractErrorMessage } from '@/lib/api'
import type { ChatMessage, Citation } from '@/lib/types'

function makeId(): string {
  return crypto.randomUUID()
}

export function ChatPanel({ knowledgeBaseId }: { knowledgeBaseId: string }) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [openCitation, setOpenCitation] = useState<Citation | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const sendMessage = useSendChatMessage()

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault()
    const question = input.trim()
    if (!question || sendMessage.isPending) return

    const userMessage: ChatMessage = { id: makeId(), role: 'user', content: question }
    const pendingId = makeId()
    setMessages((prev) => [...prev, userMessage, { id: pendingId, role: 'assistant', content: '', pending: true }])
    setInput('')

    sendMessage.mutate(
      { question, knowledgeBaseId },
      {
        onSuccess: (response) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === pendingId
                ? {
                    id: pendingId,
                    role: 'assistant',
                    content: response.answer,
                    citations: response.citations,
                    modelUsed: response.model_used,
                    contextFound: response.context_found,
                  }
                : m,
            ),
          )
        },
        onError: (error) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === pendingId
                ? { id: pendingId, role: 'assistant', content: '', error: extractErrorMessage(error) }
                : m,
            ),
          )
        },
      },
    )
  }

  return (
    <div className="flex h-[calc(100vh-13rem)] flex-col">
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <EmptyState
            title="Ask this knowledge base a question"
            description="Answers are grounded in your documents and cited back to the exact passage."
          />
        ) : (
          <div className="flex flex-col gap-3 pb-4">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} onOpenCitation={setOpenCitation} />
            ))}
            <div ref={scrollRef} />
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2 border-t border-ink-700 pt-3">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about this knowledge base..."
          className="flex-1 rounded-sm border border-ink-600 bg-ink-900 px-3 py-2 text-sm text-paper-100
            placeholder:text-paper-faint outline-none focus:border-gold-500/70"
        />
        <Button type="submit" isLoading={sendMessage.isPending} disabled={!input.trim()}>
          Send
        </Button>
      </form>

      {openCitation && (
        <ChunkViewerModal
          documentId={openCitation.document_id}
          filename={openCitation.document_filename}
          highlightChunkId={openCitation.chunk_id}
          onClose={() => setOpenCitation(null)}
        />
      )}
    </div>
  )
}
