import type { ChatResponse, Message } from '../types/chat'

const API_PATHS = ['/query', '/api/chat']

type ChatRequest = {
  question: string
  history: Array<Pick<Message, 'role' | 'content'>>
}

function normalizeResponse(data: unknown): ChatResponse {
  if (
    typeof data === 'object' &&
    data !== null &&
    'answer' in data &&
    typeof data.answer === 'string'
  ) {
    const sources =
      'sources' in data && Array.isArray(data.sources)
        ? data.sources.filter((source): source is string => typeof source === 'string')
        : []

    return {
      answer: data.answer,
      sources,
    }
  }

  throw new Error('The server returned an invalid response.')
}

export async function sendMessage(
  question: string,
  history: Message[],
): Promise<ChatResponse> {
  const payload: ChatRequest = {
    question,
    history: history.map(({ role, content }) => ({ role, content })),
  }

  let lastError: Error | null = null

  for (const path of API_PATHS) {
    try {
      const response = await fetch(path, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        const errorData = (await response.json().catch(() => null)) as
          | { error?: string; detail?: string }
          | null
        const detail =
          errorData?.error ?? errorData?.detail ?? `Request failed with ${response.status}.`

        throw new Error(detail)
      }

      return normalizeResponse(await response.json())
    } catch (error) {
      lastError = error instanceof Error ? error : new Error('Unknown network error.')
    }
  }

  throw lastError ?? new Error('Unable to reach the chatbot service.')
}
