import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_answer(prompt: str) -> str:
    """Generate answer using LLM (Anthropic or Azure OpenAI)"""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        return _generate_with_anthropic(prompt)
    elif provider == "azure_openai":
        return _generate_with_azure_openai(prompt)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _generate_with_anthropic(prompt: str) -> str:
    """Generate answer using Anthropic Claude"""
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        system_prompt = """You are a helpful assistant. Answer ONLY based on the document context provided.
If the answer is not in the context, respond with: 'I could not find this in the help document.'
Never make up answers. Always cite the page number of your source."""

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = message.content[0].text
        logger.info("Generated answer using Anthropic Claude")
        return answer

    except Exception as e:
        logger.error(f"Error generating answer with Anthropic: {e}")
        raise


def _generate_with_azure_openai(prompt: str) -> str:
    """Generate answer using Azure OpenAI"""
    try:
        from azure.openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-08-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

        system_prompt = """You are a helpful assistant. Answer ONLY based on the document context provided.
If the answer is not in the context, respond with: 'I could not find this in the help document.'
Never make up answers. Always cite the page number of your source."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content
        logger.info("Generated answer using Azure OpenAI")
        return answer

    except Exception as e:
        logger.error(f"Error generating answer with Azure OpenAI: {e}")
        raise
