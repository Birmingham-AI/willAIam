from typing import List, Dict

from clients import get_supabase, get_embedding


class RAGService:
    """Service for RAG operations: embedding and search"""

    async def search_meeting_notes(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search meeting notes using vector similarity via Supabase.

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            List of top matching results with scores
        """
        query_embedding = await get_embedding(query)
        supabase = await get_supabase()

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
