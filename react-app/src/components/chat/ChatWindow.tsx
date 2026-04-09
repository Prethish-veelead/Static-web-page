import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { useEffect, useEffectEvent, useRef } from 'react'
import { AppHeader } from './AppHeader'
import { InputBox } from './InputBox'
import { MessageBubble } from './MessageBubble'
import { TypingIndicator } from './TypingIndicator'
import { useChat } from '../../hooks/useChat'

const EMPTY_STATE_SUGGESTIONS = [
  'Summarize the nutrition guidance in the manual.',
  'Which sections talk about balanced meals?',
  'What are the main takeaways from this document?',
]

export function ChatWindow() {
  const {
    messages,
    isLoading,
    error,
    tokensUsed,
    tokenLimit,
    remainingTokens,
    sendMessage,
    clearChat,
    refreshTokens,
  } = useChat()
  const shouldReduceMotion = useReducedMotion()
  const endRef = useRef<HTMLDivElement | null>(null)

  const scrollToLatest = useEffectEvent(() => {
    endRef.current?.scrollIntoView({
      behavior: shouldReduceMotion ? 'auto' : 'smooth',
      block: 'end',
    })
  })

  useEffect(() => {
    refreshTokens()
  }, [refreshTokens])

  useEffect(() => {
    scrollToLatest()
  }, [messages, isLoading])

  return (
    <section className="flex min-h-[calc(100vh-3rem)] w-full flex-col overflow-hidden rounded-[32px] border border-white/70 bg-chat-panel shadow-chat backdrop-blur-xl">
      <AppHeader
        tokensUsed={tokensUsed}
        tokenLimit={tokenLimit}
        remainingTokens={remainingTokens}
        onClear={clearChat}
      />

      <div className="flex-1 overflow-y-auto px-3 py-4 sm:px-4">
        {messages.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.22 }}
            className="flex h-full min-h-[320px] flex-col items-center justify-center rounded-[28px] border border-dashed border-stone-300/80 bg-white/45 px-6 py-10 text-center"
          >
            <div className="max-w-md">
              <p className="text-sm font-semibold uppercase tracking-[0.32em] text-amber-700/70">
                Ready to search
              </p>
              <h2 className="mt-4 text-3xl font-semibold tracking-tight text-stone-900">
                Your retrieval assistant is standing by.
              </h2>
              <p className="mt-3 text-sm leading-7 text-stone-600">
                Ask clear questions, get concise answers, and review the source tags that come
                back with each response.
              </p>
              <div className="mt-6 flex flex-wrap justify-center gap-2">
                {EMPTY_STATE_SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => void sendMessage(suggestion)}
                    className="rounded-full border border-stone-300/80 bg-white/80 px-3 py-2 text-sm text-stone-700 transition hover:-translate-y-0.5 hover:border-stone-400 hover:bg-white"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          </motion.div>
        ) : (
          <div className="space-y-3">
            <AnimatePresence initial={false}>
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
              {isLoading ? <TypingIndicator key="typing-indicator" /> : null}
            </AnimatePresence>
          </div>
        )}

        {error ? (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mx-auto mt-4 max-w-xl rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700"
          >
            {error}
          </motion.div>
        ) : null}

        <div ref={endRef} />
      </div>

      <InputBox
        disabled={isLoading || remainingTokens <= 0}
        remainingTokens={remainingTokens}
        onSubmit={sendMessage}
      />
    </section>
  )
}
