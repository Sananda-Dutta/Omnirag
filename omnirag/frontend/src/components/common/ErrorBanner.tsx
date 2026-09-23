export function ErrorBanner({ message }: { message: string }) {
  return (
    <div
      role="alert"
      className="rounded-sm border border-rust-500/30 bg-rust-500/10 px-3 py-2 text-sm text-rust-400"
    >
      {message}
    </div>
  )
}
