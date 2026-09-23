import { type ButtonHTMLAttributes, forwardRef } from 'react'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  isLoading?: boolean
}

const VARIANT_CLASSES: Record<Variant, string> = {
  primary:
    'bg-gold-500 text-ink-950 hover:bg-gold-400 disabled:bg-gold-500/40 disabled:text-ink-950/50',
  secondary:
    'bg-ink-800 text-paper-100 border border-ink-600 hover:border-gold-500/60 disabled:opacity-50',
  ghost: 'text-paper-muted hover:text-paper-100 hover:bg-ink-800 disabled:opacity-40',
  danger: 'bg-rust-500/15 text-rust-400 border border-rust-500/30 hover:bg-rust-500/25 disabled:opacity-50',
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = 'primary', isLoading, disabled, className = '', children, ...rest }, ref) => (
    <button
      ref={ref}
      disabled={disabled || isLoading}
      className={`inline-flex items-center justify-center gap-2 rounded-sm px-4 py-2 text-sm font-medium
        transition-colors duration-150 cursor-pointer disabled:cursor-not-allowed
        ${VARIANT_CLASSES[variant]} ${className}`}
      {...rest}
    >
      {isLoading && (
        <span
          className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent"
          aria-hidden="true"
        />
      )}
      {children}
    </button>
  ),
)
Button.displayName = 'Button'
