from os import getenv
from fastapi import HTTPException
from supabase._async.client import create_client as create_async_client, AsyncClient

SUPABASE_URL = getenv("SUPABASE_URL")
SUPABASE_KEY = getenv("SUPABASE_KEY")

# Async Supabase client (lazy initialized)
_supabase_client: AsyncClient | None = None


async def get_supabase() -> AsyncClient:
    """Get or create async Supabase client"""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = await create_async_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


def check_supabase_configured():
    """Raise HTTPException if Supabase is not configured"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(
            status_code=500,
            detail="Supabase credentials not configured"
        )
