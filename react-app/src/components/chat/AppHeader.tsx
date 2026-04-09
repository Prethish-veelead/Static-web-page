import { memo } from 'react'

type AppHeaderProps = {
  tokensUsed: number
  tokenLimit: number
  remainingTokens: number
  onClear: () => void
}

export const AppHeader = memo(function AppHeader({
  tokensUsed,
  tokenLimit,
  remainingTokens,
  onClear,
}: AppHeaderProps) {
  const usageRatio = Math.min(100, Math.round((tokensUsed / tokenLimit) * 100))

  return (
    <header className="border-b border-chat-border px-4 py-4 sm:px-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-amber-700/70">
            RAG Assistant
          </p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight text-stone-900">
            Ask your knowledge base anything
          </h1>
          <p className="mt-2 text-sm text-stone-600">
            Fast answers with source context, daily usage controls, and a clean chat flow.
          </p>
        </div>

        <button
          type="button"
          onClick={onClear}
          className="rounded-full border border-stone-300/70 bg-white/75 px-4 py-2 text-sm font-medium text-stone-700 transition hover:-translate-y-0.5 hover:border-stone-400 hover:bg-white"
        >
          Clear chat
        </button>
      </div>

      <div className="mt-5 space-y-2">
        <div className="flex items-center justify-between gap-4 text-sm text-stone-600">
          <span>Tokens: {tokensUsed} / {tokenLimit}</span>
          <span>{remainingTokens} left today</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-stone-200/80">
          <div
            className="h-full rounded-full bg-gradient-to-r from-amber-500 via-orange-500 to-rose-500 transition-[width] duration-300"
            style={{ width: `${usageRatio}%` }}
          />
        </div>
      </div>
    </header>
  )
})
