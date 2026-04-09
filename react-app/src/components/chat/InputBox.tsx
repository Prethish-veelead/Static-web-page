import { memo, useDeferredValue, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { useDebouncedValue } from '../../hooks/useDebouncedValue'

type InputBoxProps = {
  disabled: boolean
  remainingTokens: number
  onSubmit: (value: string) => Promise<void>
}

function estimateTokens(text: string) {
  const words = text.trim().split(/\s+/).filter(Boolean).length
  return Math.ceil(words * 1.3)
}

export const InputBox = memo(function InputBox({
  disabled,
  remainingTokens,
  onSubmit,
}: InputBoxProps) {
  const [value, setValue] = useState('')
  const deferredValue = useDeferredValue(value)
  const debouncedValue = useDebouncedValue(deferredValue)
  const estimatedTokens = estimateTokens(debouncedValue)
  const trimmedValue = value.trim()
  const helperText = useMemo(() => {
    if (!trimmedValue) {
      return 'Ask about the indexed document...'
    }

    return `${estimatedTokens} estimated tokens for this prompt${remainingTokens >= 0 ? ` | ${remainingTokens} left today` : ''}`
  }, [estimatedTokens, remainingTokens, trimmedValue])

  async function handleSubmit() {
    if (!trimmedValue || disabled) {
      return
    }

    const nextValue = trimmedValue
    setValue('')
    await onSubmit(nextValue)
  }

  return (
    <div className="border-t border-chat-border bg-white/65 px-3 pb-3 pt-3 backdrop-blur sm:px-4 sm:pb-4">
      <div className="rounded-[28px] border border-stone-200/80 bg-white p-2 shadow-[0_10px_30px_rgba(120,53,15,0.08)]">
        <div className="flex items-end gap-2">
          <textarea
            value={value}
            rows={1}
            placeholder="Ask a question about your documents..."
            disabled={disabled}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                void handleSubmit()
              }
            }}
            className="max-h-36 min-h-[52px] flex-1 resize-none bg-transparent px-3 py-3 text-[15px] leading-6 text-stone-900 outline-none placeholder:text-stone-400 disabled:cursor-not-allowed disabled:opacity-60"
          />

          <motion.button
            type="button"
            whileHover={disabled ? undefined : { scale: 1.02, y: -1 }}
            whileTap={disabled ? undefined : { scale: 0.98 }}
            onClick={() => void handleSubmit()}
            disabled={disabled || !trimmedValue}
            className="mb-1 inline-flex h-11 min-w-24 items-center justify-center rounded-full bg-stone-900 px-4 text-sm font-semibold text-white transition disabled:cursor-not-allowed disabled:bg-stone-300"
          >
            Send
          </motion.button>
        </div>

        <p className="px-3 pb-1 pt-2 text-xs text-stone-500">{helperText}</p>
      </div>
    </div>
  )
})
