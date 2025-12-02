from dotenv import load_dotenv
from os.path import join, dirname
from os import getenv
from openai import OpenAI
from supabase import create_client
from typing import List, Dict

# Load environment variables
load_dotenv(join(dirname(dirname(dirname(__file__))), ".env"))

OPENAI_API_KEY = getenv("OPENAI_API_KEY")
SUPABASE_URL = getenv("SUPABASE_URL")
SUPABASE_KEY = getenv("SUPABASE_KEY")


class RAGService:
    """Service for RAG operations: embedding and search"""

    def __init__(self):
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text string"""
        resp = self.openai.embeddings.create(model="text-embedding-3-small", input=text)
        return resp.data[0].embedding

    def search_meeting_notes(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search meeting notes using vector similarity via Supabase.

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            List of top matching results with scores
        """
        query_embedding = self.get_embedding(query)

        # Use Supabase RPC for vector similarity search
        results = self.supabase.rpc(
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

