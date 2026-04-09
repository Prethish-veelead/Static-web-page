from typing import List, Dict, Any


SYSTEM_PROMPT = """You are a helpful assistant. Answer ONLY based on the document context provided.
If the answer is not in the context, respond with: 'I could not find this in the help document.'
Never make up answers. Always cite the page number of your source."""


def format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks as context with page numbers"""
    context_parts = []
    for chunk in retrieved_chunks:
        text = chunk.get("text", "")
        page = chunk.get("page", "?")
        context_parts.append(f"[Page {page}]\n{text}")

    return "\n\n".join(context_parts)


def format_history(history: List[Dict[str, str]]) -> str:
    """Format conversation history"""
    if not history:
        return ""

    history_parts = []
    for msg in history:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        history_parts.append(f"{role}: {content}")

    return "\n".join(history_parts)


def build_prompt(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    history: List[Dict[str, str]],
) -> str:
    """Build complete prompt with context, history, and question"""

    context = format_context(retrieved_chunks)
    history_text = format_history(history)

    user_prompt = f"""Context from help document:
{context}

Conversation history:
{history_text}

Question: {question}"""

    return user_prompt
