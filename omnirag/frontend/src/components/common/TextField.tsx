import { type InputHTMLAttributes, forwardRef } from 'react'

interface TextFieldProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
  hint?: string
}

export const TextField = forwardRef<HTMLInputElement, TextFieldProps>(
  ({ label, hint, id, className = '', ...rest }, ref) => (
    <label className="flex flex-col gap-1.5" htmlFor={id}>
      {label && <span className="text-sm text-paper-muted">{label}</span>}
      <input
        ref={ref}
        id={id}
        className={`rounded-sm border border-ink-600 bg-ink-900 px-3 py-2 text-sm text-paper-100
          placeholder:text-paper-faint outline-none transition-colors
          focus:border-gold-500/70 ${className}`}
        {...rest}
      />
      {hint && <span className="text-xs text-paper-faint">{hint}</span>}
    </label>
  ),
)
TextField.displayName = 'TextField'
