"""
Save YouTube transcriptions to Supabase.

Usage:
    python -m backend.actions.save_embed_supabase --url "VIDEO_URL" --session "Nov 2024 Birmingham AI Meetup"
"""

import argparse
from os import getenv
from os.path import join, dirname

from dotenv import load_dotenv
from supabase import create_client

from backend.actions.transcribe_youtube import YouTubeTranscriber

# Load environment variables
load_dotenv(join(dirname(dirname(dirname(__file__))), ".env"))

SUPABASE_URL = getenv("SUPABASE_URL")
SUPABASE_KEY = getenv("SUPABASE_KEY")


def save_to_supabase(url: str, session_info: str) -> None:
    """
    Transcribe YouTube video and save embeddings to Supabase.

    Args:
        url: YouTube video URL or video ID
        session_info: Description of the session
    """
    # Extract video ID first to check if already processed
    video_id = YouTubeTranscriber.extract_video_id(url)
    if not video_id:
        print(f"Error: Could not extract video ID from: {url}")
        return

    # Connect to Supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Check if already processed
    existing = supabase.table("sources").select("id").eq(
        "source_type", "youtube"
    ).eq("source_id", video_id).execute()

    if existing.data:
        print(f"Already processed, skipping: {video_id}")
        return

    # Transcribe and embed (skip local JSON file by default)
    print(f"Processing YouTube video: {video_id}")
    transcriber = YouTubeTranscriber()
    chunks = transcriber.transcribe(url, session_info, save_local=False)

    # Insert source record
    source_result = supabase.table("sources").insert({
        "source_type": "youtube",
        "source_id": video_id,
        "session_info": session_info,
        "chunk_count": len(chunks)
    }).execute()

    source_uuid = source_result.data[0]["id"]

    # Insert embeddings
    for chunk in chunks:
        supabase.table("embeddings").insert({
            "source_id": source_uuid,
            "text": chunk["text"],
            "timestamp": chunk["timestamp"],
            "embedding": chunk["embedding"]
        }).execute()

    print(f"Saved {len(chunks)} chunks to Supabase")


def main():
    parser = argparse.ArgumentParser(
        description="Save YouTube transcriptions to Supabase"
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="YouTube video URL or video ID"
    )
    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Session info (e.g., 'Nov 2024 Birmingham AI Meetup')"
    )

    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set in .env")
        raise SystemExit(1)

    try:
        save_to_supabase(args.url, args.session)
    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
