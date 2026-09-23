import { useRef, useState } from 'react'

import { Button } from '@/components/common/Button'
import { ErrorBanner } from '@/components/common/ErrorBanner'
import { TextField } from '@/components/common/TextField'
import { useIngestUrl, useUploadDocument } from '@/hooks/useDocuments'
import { extractErrorMessage } from '@/lib/api'

export function UploadPanel({ knowledgeBaseId }: { knowledgeBaseId: string }) {
  const [mode, setMode] = useState<'file' | 'url'>('file')
  const [url, setUrl] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const uploadMutation = useUploadDocument(knowledgeBaseId)
  const ingestUrlMutation = useIngestUrl(knowledgeBaseId)

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0]
    if (!file) return
    uploadMutation.mutate(file)
  }

  const handleUrlSubmit = (event: React.FormEvent) => {
    event.preventDefault()
    if (!url.trim()) return
    ingestUrlMutation.mutate(url.trim(), {
      onSuccess: () => setUrl(''),
    })
  }

  return (
    <div className="rounded-sm border border-ink-700 bg-ink-900/60 p-4">
      <div className="mb-3 flex gap-1 text-sm">
        <button
          type="button"
          onClick={() => setMode('file')}
          className={`rounded-sm px-3 py-1.5 transition-colors ${
            mode === 'file' ? 'bg-ink-800 text-paper-100' : 'text-paper-muted hover:text-paper-100'
          }`}
        >
          Upload a file
        </button>
        <button
          type="button"
          onClick={() => setMode('url')}
          className={`rounded-sm px-3 py-1.5 transition-colors ${
            mode === 'url' ? 'bg-ink-800 text-paper-100' : 'text-paper-muted hover:text-paper-100'
          }`}
        >
          Add from a URL
        </button>
      </div>

      {mode === 'file' ? (
        <div
          onDragOver={(e) => {
            e.preventDefault()
            setIsDragging(true)
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setIsDragging(false)
            handleFiles(e.dataTransfer.files)
          }}
          className={`flex flex-col items-center justify-center gap-2 rounded-sm border border-dashed px-6 py-8 text-center transition-colors ${
            isDragging ? 'border-gold-500 bg-gold-500/5' : 'border-ink-600'
          }`}
        >
          <p className="text-sm text-paper-muted">
            Drop a PDF, DOCX, TXT, Markdown, or image file here, or
          </p>
          <Button
            type="button"
            variant="secondary"
            isLoading={uploadMutation.isPending}
            onClick={() => fileInputRef.current?.click()}
          >
            Choose a file
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>
      ) : (
        <form onSubmit={handleUrlSubmit} className="flex flex-col gap-3">
          <TextField
            type="url"
            placeholder="https://example.com/article"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            required
          />
          <Button type="submit" isLoading={ingestUrlMutation.isPending} className="self-start">
            Fetch and add
          </Button>
        </form>
      )}

      {uploadMutation.isError && (
        <div className="mt-3">
          <ErrorBanner message={extractErrorMessage(uploadMutation.error)} />
        </div>
      )}
      {ingestUrlMutation.isError && (
        <div className="mt-3">
          <ErrorBanner message={extractErrorMessage(ingestUrlMutation.error)} />
        </div>
      )}
    </div>
  )
}
