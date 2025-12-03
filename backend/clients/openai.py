from os import getenv
from openai import AsyncOpenAI

OPENAI_API_KEY = getenv("OPENAI_API_KEY")

# Shared async OpenAI client (lazy initialized)
_openai_client: AsyncOpenAI | None = None


def get_openai() -> AsyncOpenAI:
    """Get or create async OpenAI client"""
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


async def get_embedding(text: str) -> list[float]:
    """Get embedding for a text string using text-embedding-3-small"""
    client = get_openai()
    resp = await client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding
