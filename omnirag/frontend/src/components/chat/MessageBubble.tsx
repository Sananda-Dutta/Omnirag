import ReactMarkdown from 'react-markdown'

import { CitationChip } from '@/components/chat/CitationChip'
import { Spinner } from '@/components/common/Spinner'
import type { ChatMessage, Citation } from '@/lib/types'

export function MessageBubble({
  message,
  onOpenCitation,
}: {
  message: ChatMessage
  onOpenCitation: (citation: Citation) => void
}) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[75ch] rounded-sm px-4 py-3 text-sm leading-relaxed ${
          isUser ? 'bg-gold-500/10 text-paper-100' : 'bg-ink-850 text-paper-100'
        }`}
      >
        {message.pending ? (
          <Spinner label="Thinking..." />
        ) : message.error ? (
          <p className="text-rust-400">{message.error}</p>
        ) : (
          <div className="prose-invert max-w-none [&_a]:text-gold-400 [&_p]:my-1.5 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}

        {!isUser && !message.pending && message.contextFound === false && (
          <p className="mt-1 font-mono text-xs text-paper-faint">
            No matching content found in this knowledge base.
          </p>
        )}

        {message.citations && message.citations.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {message.citations.map((citation, i) => (
              <CitationChip
                key={citation.chunk_id}
                citation={citation}
                index={i + 1}
                onClick={() => onOpenCitation(citation)}
              />
            ))}
          </div>
        )}

        {!isUser && message.modelUsed && !message.pending && (
          <p className="mt-2 font-mono text-[11px] text-paper-faint">{message.modelUsed}</p>
        )}
      </div>
    </div>
  )
}
