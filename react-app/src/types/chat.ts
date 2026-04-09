export type MessageRole = 'user' | 'assistant'

export type Message = {
  id: string
  role: MessageRole
  content: string
  sources?: string[]
}

export type ChatResponse = {
  answer: string
  sources: string[]
}

export type TokenUsage = {
  tokensUsed: number
  lastReset: number
}
