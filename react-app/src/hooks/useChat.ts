import { startTransition, useCallback, useMemo, useState } from 'react'
import { sendMessage as sendChatMessage } from '../services/chatService'
import type { Message } from '../types/chat'

const DAILY_TOKEN_LIMIT = 10_000
const RESET_INTERVAL_MS = 24 * 60 * 60 * 1000
const STORAGE_KEY = 'rag-chat-token-usage'

type TokenStore = {
  tokensUsed: number
  lastReset: number
}

function estimateTokens(text: string) {
  const words = text.trim().split(/\s+/).filter(Boolean).length
  return Math.ceil(words * 1.3)
}

function createMessage(role: Message['role'], content: string, sources?: string[]): Message {
  return {
    id: `${role}-${crypto.randomUUID()}`,
    role,
    content,
    sources,
  }
}

function readTokenStore(): TokenStore {
  const now = Date.now()

  if (typeof window === 'undefined') {
    return { tokensUsed: 0, lastReset: now }
  }

  const rawValue = window.localStorage.getItem(STORAGE_KEY)

  if (!rawValue) {
    return { tokensUsed: 0, lastReset: now }
  }

  try {
    const parsed = JSON.parse(rawValue) as Partial<TokenStore>

    if (
      typeof parsed.tokensUsed !== 'number' ||
      typeof parsed.lastReset !== 'number' ||
      now - parsed.lastReset >= RESET_INTERVAL_MS
    ) {
      return { tokensUsed: 0, lastReset: now }
    }

    return {
      tokensUsed: parsed.tokensUsed,
      lastReset: parsed.lastReset,
    }
  } catch {
    return { tokensUsed: 0, lastReset: now }
  }
}

function persistTokenStore(store: TokenStore) {
  if (typeof window === 'undefined') {
    return
  }

  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(store))
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [tokenStore, setTokenStore] = useState<TokenStore>(() => {
    const store = readTokenStore()
    persistTokenStore(store)
    return store
  })

  const syncTokenStore = useCallback(() => {
    const nextStore = readTokenStore()
    persistTokenStore(nextStore)
    setTokenStore(nextStore)
    return nextStore
  }, [])

  const sendMessage = useCallback(
    async (question: string) => {
      const trimmedQuestion = question.trim()

      if (!trimmedQuestion || isLoading) {
        return
      }

      const refreshedStore = syncTokenStore()
      const estimatedQuestionTokens = estimateTokens(trimmedQuestion)

      if (refreshedStore.tokensUsed + estimatedQuestionTokens > DAILY_TOKEN_LIMIT) {
        setError('Daily token limit reached. Please try again tomorrow.')
        return
      }

      setError(null)
      setIsLoading(true)

      const userMessage = createMessage('user', trimmedQuestion)
      const nextHistory = [...messages, userMessage]
      setMessages(nextHistory)

      try {
        const response = await sendChatMessage(trimmedQuestion, messages)
        const answerTokens = estimateTokens(response.answer)
        const nextStore = {
          tokensUsed: refreshedStore.tokensUsed + estimatedQuestionTokens + answerTokens,
          lastReset: refreshedStore.lastReset,
        }

        persistTokenStore(nextStore)
        setTokenStore(nextStore)

        startTransition(() => {
          setMessages((currentMessages) => [
            ...currentMessages,
            createMessage('assistant', response.answer, response.sources),
          ])
        })
      } catch (sendError) {
        const message =
          sendError instanceof Error
            ? sendError.message
            : 'Something went wrong while reaching the assistant.'

        setError(message)
      } finally {
        setIsLoading(false)
      }
    },
    [isLoading, messages, syncTokenStore],
  )

  const clearChat = useCallback(() => {
    setMessages([])
    setError(null)
    const refreshedStore = syncTokenStore()
    setTokenStore(refreshedStore)
  }, [syncTokenStore])

  const remainingTokens = useMemo(
    () => Math.max(0, DAILY_TOKEN_LIMIT - tokenStore.tokensUsed),
    [tokenStore.tokensUsed],
  )

  return {
    messages,
    isLoading,
    error,
    tokensUsed: Math.min(tokenStore.tokensUsed, DAILY_TOKEN_LIMIT),
    tokenLimit: DAILY_TOKEN_LIMIT,
    remainingTokens,
    sendMessage,
    clearChat,
    refreshTokens: syncTokenStore,
  }
}
