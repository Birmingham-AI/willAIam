from pandas import read_json
from dotenv import load_dotenv
from os.path import join, dirname
from os import getenv
from openai import OpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner, set_tracing_disabled
import numpy as np
from pathlib import Path
import re
from typing import List, Dict, AsyncGenerator
import json

# Load environment variables
load_dotenv(join(dirname(dirname(dirname(__file__))), ".env"))

# Disable tracing for ZDR (Zero Data Retention) organizations
set_tracing_disabled(True)

OPENAI_API_KEY = getenv("OPENAI_API_KEY")

# In Docker, /app is the backend dir, embeddings are mounted at /app/embeddings
# Outside Docker, we need to go up to project root
if Path("/app/embeddings").exists():
    # Running in Docker container
    EMBEDDINGS_DIR = Path("/app/embeddings")
else:
    # Running locally - go up to project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

BUNDLED_DIR = EMBEDDINGS_DIR / "bundled"
BUNDLE_FILE_PATTERN = re.compile(r"^bundle-(\d+)\.json$")


class RAGService:
    """Service for RAG operations: search and answer synthesis"""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def get_embeddings_file(self) -> Path:
        """Get the latest bundled embeddings file"""
        existing = [BUNDLE_FILE_PATTERN.match(path.name) for path in BUNDLED_DIR.glob("bundle-*.json")]
        indices = [int(match.group(1)) for match in existing if match]
        bundle_path = BUNDLED_DIR / f"bundle-{max(indices, default=0)}.json"
        if not bundle_path.exists():
            raise FileNotFoundError(
                "No bundled embeddings found. Run actions/bundle.py to create one."
            )
        return bundle_path

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text string"""
        resp = self.client.embeddings.create(model="text-embedding-3-large", input=text)
        return resp.data[0].embedding

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def search_meeting_notes(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search meeting notes using vector similarity

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            List of top matching results with scores
        """
        # Get query embedding
        query_embedding = self.get_embedding(query)

        # Load embedded meeting notes
        meeting_data = read_json(self.get_embeddings_file())

        # Calculate similarity scores
        results = []
        for index, row in meeting_data.iterrows():
            similarity = self.cosine_similarity(query_embedding, row['embedding'])
            results.append({
                'slide': row['slide'],
                'year': row['year'],
                'month': row['month'],
                'text': row['text'],
                'score': float(similarity)
            })

        # Sort by similarity score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:top_k]

    async def synthesize_answer_streaming(
        self, question: str, results: List[Dict]
    ) -> AsyncGenerator[str, None]:
        """
        Stream a conversational answer using OpenAI Agents SDK

        Args:
            question: The user's question
            results: Search results from vector similarity

        Yields:
            Text chunks as they are generated
        """
        if not results:
            yield "I couldn't find anything in the meeting notes that answers that yet."
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
                yield event.data.delta

    async def synthesize_answer_non_streaming(
        self, question: str, results: List[Dict]
    ) -> str:
        """
        Generate a conversational answer without streaming (for REST API)

        Args:
            question: The user's question
            results: Search results from vector similarity

        Returns:
            Complete answer as a string
        """
        if not results:
            return "I couldn't find anything in the meeting notes that answers that yet."

        context_lines = []
        for idx, result in enumerate(results, start=1):
            context_lines.append(
                f"{idx}. Year: {result['year']}, Month: {result['month']}, Slide: {result['slide']}\n"
                f"Summary: {result['text']}"
            )

        system_prompt = (
            "You read meeting-note snippets and answer the user's question. "
            "Be conversational but concise. Cite the year and month whenever you "
            "mention a supporting point (format 'Discussed in YEAR/MONTH'). If the "
            "notes don't address the question, say that plainly."
        )

        user_content = (
            f"Question: {question}\n\nRelevant notes:\n" + "\n\n".join(context_lines)
        )

        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )

        return completion.choices[0].message.content.strip()
