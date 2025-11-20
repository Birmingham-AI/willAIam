from pandas import read_json
from dotenv import load_dotenv
from os.path import join, dirname
from os import getenv
from openai import OpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner, set_tracing_disabled
import numpy as np
import sys
import asyncio
from pathlib import Path
import re
from typing import List, Dict

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(join(dirname(dirname(__file__)), ".env"))

# Disable tracing for ZDR (Zero Data Retention) organizations
set_tracing_disabled(True)

OPENAI_API_KEY = getenv("OPENAI_API_KEY")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
BUNDLED_DIR = EMBEDDINGS_DIR / "bundled"
BUNDLE_FILE_PATTERN = re.compile(r"^bundle-(\d+)\.json$")

client = OpenAI(api_key=OPENAI_API_KEY)

def get_embeddings_file():
    existing = [BUNDLE_FILE_PATTERN.match(path.name) for path in BUNDLED_DIR.glob("bundle-*.json")]
    indices = [int(match.group(1)) for match in existing if match]
    bundle_path = BUNDLED_DIR / f"bundle-{max(indices, default=0)}.json"
    if not bundle_path.exists():
        raise FileNotFoundError(
            "No bundled embeddings found. Run actions/bundle.py to create one."
        )
    return bundle_path


def get_embedding(text):
    resp = client.embeddings.create(model="text-embedding-3-large", input=text)
    return resp.data[0].embedding

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def search_meeting_notes(query, top_k=5):
    """
    Search meeting notes using vector similarity
    
    Args:
        query (str): The search query
        top_k (int): Number of top results to return
        
    Returns:
        list: Top matching results with scores
    """
    # Get query embedding
    query_embedding = get_embedding(query)
    
    # Load embedded meeting notes
    meeting_data = read_json(get_embeddings_file())
    
    # Calculate similarity scores
    results = []
    for index, row in meeting_data.iterrows():
        similarity = cosine_similarity(query_embedding, row['embedding'])
        results.append({
            'slide': row['slide'],
            'year': row['year'],
            'month': row['month'],
            'text': row['text'],
            'score': similarity
        })
    
    # Sort by similarity score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results[:top_k]


async def synthesize_answer_streaming(question: str, results: List[Dict]) -> None:
    """Use OpenAI Agents SDK to stream a conversational answer grounded in retrieved notes."""
    if not results:
        print("I couldn't find anything in the meeting notes that answers that yet.")
        return

    context_lines = []
    for idx, result in enumerate(results, start=1):
        context_lines.append(
            f"{idx}. Year: {result['year']}, Month: {result['month']}, Slide: {result['slide']}\n"
            f"Summary: {result['text']}"
        )

    instructions = (
        "You read meeting-note snippets and answer the user's question. "
        "Be conversational but concise. Cite the year and month whenever you "
        "mention a supporting point (format 'Discussed in YEAR/MONTH'). If the "
        "notes don't address the question, say that plainly."
    )

    user_content = (
        f"Question: {question}\n\nRelevant notes:\n" + "\n\n".join(context_lines)
    )

    # Create agent with instructions
    agent = Agent(
        name="MeetingNotesAssistant",
        instructions=instructions,
        model="gpt-4o-mini",
    )

    # Run agent in streaming mode
    result = Runner.run_streamed(agent, input=user_content)

    # Stream token-by-token output
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

    # Add newline after streaming completes
    print()

async def main():
    """Main async function to handle the Q&A flow with streaming."""
    print("Meeting Notes Search")
    print("=" * 50)

    query = input("\nEnter your question: ")

    print(f"\nSearching for: {query}\n")
    try:
        results = search_meeting_notes(query)
    except FileNotFoundError as exc:
        print(str(exc))
        raise SystemExit(1)

    print("Answer:")
    print("-" * 50)

    # Stream the answer
    await synthesize_answer_streaming(query, results)

    print("\nSupporting results:")
    print("-" * 50)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [Score: {result['score']:.4f}]")
        print(f"   Slide: {result['slide']}")
        print(f"   Year: {result['year']}")
        print(f"   Month: {result['month']}")
        print(f"   Text: {result['text']}")

# Main execution
if __name__ == "__main__":
    asyncio.run(main())

