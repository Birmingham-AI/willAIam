from dotenv import load_dotenv
from os.path import join, dirname
from os import getenv
from openai import AsyncOpenAI
from supabase._async.client import create_client as create_async_client, AsyncClient
from typing import List, Dict

# Load environment variables
load_dotenv(join(dirname(dirname(dirname(__file__))), ".env"))

OPENAI_API_KEY = getenv("OPENAI_API_KEY")
SUPABASE_URL = getenv("SUPABASE_URL")
SUPABASE_KEY = getenv("SUPABASE_KEY")


class RAGService:
    """Service for RAG operations: embedding and search"""

    def __init__(self):
        self.openai = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self._supabase: AsyncClient | None = None

    async def _get_supabase(self) -> AsyncClient:
        """Get or create async Supabase client"""
        if self._supabase is None:
            self._supabase = await create_async_client(SUPABASE_URL, SUPABASE_KEY)
        return self._supabase

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text string"""
        resp = await self.openai.embeddings.create(model="text-embedding-3-small", input=text)
        return resp.data[0].embedding

    async def search_meeting_notes(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search meeting notes using vector similarity via Supabase.

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            List of top matching results with scores
        """
        query_embedding = await self.get_embedding(query)
        supabase = await self._get_supabase()

        # Use Supabase RPC for vector similarity search
        results = await supabase.rpc(
            "match_embeddings",
            {
                "query_embedding": query_embedding,
                "match_count": top_k
            }
        ).execute()

        return [
            {
                "text": row["text"],
                "timestamp": row["timestamp"],
                "session_info": row["session_info"],
                "score": row["similarity"]
            }
            for row in results.data
        ]

