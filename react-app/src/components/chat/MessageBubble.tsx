import { memo } from 'react'
import { motion } from 'framer-motion'
import type { Message } from '../../types/chat'

type MessageBubbleProps = {
  message: Message
}

export const MessageBubble = memo(function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  return (
    <motion.article
      layout
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.18, ease: 'easeOut' }}
      className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      <div
        className={[
          'max-w-[88%] rounded-3xl px-4 py-3 shadow-sm sm:max-w-[80%]',
          isUser
            ? 'rounded-br-lg bg-chat-user text-white'
            : 'rounded-bl-lg border border-stone-200/80 bg-chat-assistant text-chat-ink',
        ].join(' ')}
      >
        <p className="whitespace-pre-wrap text-[15px] leading-7">{message.content}</p>

        {!isUser && message.sources && message.sources.length > 0 ? (
          <div className="mt-3 flex flex-wrap gap-2">
            {message.sources.map((source) => (
              <span
                key={source}
                className="rounded-full bg-amber-50 px-2.5 py-1 text-xs font-medium text-amber-700"
              >
                {source}
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </motion.article>
  )
})
