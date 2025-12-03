"""
Process PDF slides and create embeddings for RAG.

Usage (CLI):
    python -m backend.actions.process_slides --pdf "slides/presentation.pdf" --session "Nov 2024 Birmingham AI Meetup"

Usage (Python):
    from backend.actions.process_slides import SlideProcessor

    processor = SlideProcessor()
    result = await processor.process("slides/presentation.pdf", session_info="Nov 2024 Birmingham AI Meetup")
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from os.path import join, dirname

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables (for CLI usage)
load_dotenv(join(dirname(dirname(dirname(__file__))), ".env"))

from clients import get_embedding

EMBEDDINGS_DIR = "embeddings"

# OpenAI model for slide analysis
MODEL = "gpt-4.1-mini"

# The Prompt for slide analysis
ANALYSIS_PROMPT = """
I want key points and topics from this slide.
Prefer information especially around AI, Machine Learning, the future of work, and news or updates about Birmingham AI.
Give them to me as JSON with the following fields:
- "slide_title": the title of the slide
- "key_points": an array of strings, each being a key point or topic from the slide

Ensure that the JSON is valid and well-formed.
"""


class SlideProcessor:
    """Process PDF slides and create embeddings for RAG."""

    def __init__(self):
        self._openai_client: AsyncOpenAI | None = None

    def _get_openai(self) -> AsyncOpenAI:
        """Get or create async OpenAI client for analysis."""
        if self._openai_client is None:
            from os import getenv
            self._openai_client = AsyncOpenAI(api_key=getenv("OPENAI_API_KEY"))
        return self._openai_client

    async def _analyze_slide(self, text: str, pdf_filename: str, page_num: int) -> dict | None:
        """Analyze slide text using OpenAI and return structured content."""
        client = self._get_openai()

        try:
            response = await client.chat.completions.create(
                model=MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helper that analyzes slide text and outputs strict JSON."
                    },
                    {
                        "role": "user",
                        "content": f"{ANALYSIS_PROMPT}\n\nFILE NAME: {pdf_filename}\nPAGE: {page_num}\n\nSLIDE TEXT:\n{text}"
                    },
                ]
            )

            return json.loads(response.choices[0].message.content)

        except json.JSONDecodeError:
            return None

    def _extract_text_from_analysis(self, analysis: dict, raw_text: str) -> str:
        """Extract readable text from analysis for embedding."""
        text_parts = []

        if analysis:
            if "slide_title" in analysis and analysis["slide_title"]:
                text_parts.append(analysis["slide_title"])
            if "key_points" in analysis and isinstance(analysis["key_points"], list):
                for point in analysis["key_points"]:
                    if isinstance(point, str):
                        text_parts.append(point)

        # Fallback to raw text if analysis didn't yield usable content
        return "\n".join(text_parts) if text_parts else raw_text

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text using shared OpenAI client."""
        return await get_embedding(text)

    async def _process_document(
        self,
        doc: fitz.Document,
        pdf_filename: str,
        session_info: str
    ) -> list[dict]:
        """
        Process a fitz document and create embeddings.

        Args:
            doc: PyMuPDF document object
            pdf_filename: Name of the PDF file (for logging/metadata)
            session_info: Description of the session

        Returns:
            List of embedded slide chunks
        """
        embedded_chunks = []
        total_pages = len(doc)
        print(f"Found {total_pages} pages")

        for page_num, page in enumerate(doc, start=1):
            start_time = time.time()

            # Extract text
            raw_text = page.get_text()

            # Skip empty pages
            if not raw_text.strip():
                print(f"  Skipping Page {page_num} (Empty)")
                continue

            print(f"  Processing Page {page_num}...", end=" ", flush=True)

            # Analyze slide content
            analysis = await self._analyze_slide(raw_text, pdf_filename, page_num)

            # Extract text for embedding
            text = self._extract_text_from_analysis(analysis, raw_text)

            # Create embedding
            embedding = await self._get_embedding(text)

            embedded_chunks.append({
                "session_info": session_info,
                "text": text,
                "timestamp": f"Slide {page_num}",
                "embedding": embedding
            })

            elapsed = time.time() - start_time
            print(f"Done ({elapsed:.2f}s)")

        return embedded_chunks

    async def process_from_bytes(
        self,
        pdf_bytes: bytes,
        filename: str,
        session_info: str
    ) -> list[dict]:
        """
        Process PDF from bytes in memory and create embeddings.

        Args:
            pdf_bytes: PDF file content as bytes
            filename: Name of the PDF file (for logging/metadata)
            session_info: Description of the session

        Returns:
            List of embedded slide chunks
        """
        print(f"Processing: {filename}")

        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                return await self._process_document(doc, filename, session_info)
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise

    async def process(
        self,
        pdf_path: str,
        session_info: str,
        output_filename: str = None,
        save_local: bool = True
    ) -> list[dict]:
        """
        Process PDF slides from file path and create embeddings.

        Args:
            pdf_path: Path to the PDF file
            session_info: Description of the session (e.g., "Nov 2024 Birmingham AI Meetup")
            output_filename: Optional custom output filename
            save_local: Whether to save JSON file locally (default: True)

        Returns:
            List of embedded slide chunks
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pdf_filename = pdf_path.name

        try:
            with fitz.open(str(pdf_path)) as doc:
                embedded_chunks = await self._process_document(doc, pdf_filename, session_info)
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise

        # Save to embeddings directory (optional)
        if save_local:
            if output_filename is None:
                output_filename = f"slides-{pdf_path.stem}.json"

            if not output_filename.endswith(".json"):
                output_filename += ".json"

            output_path = os.path.join(EMBEDDINGS_DIR, output_filename)
            os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)

            print(f"\nSaved to: {output_path}")

        print(f"Total slides processed: {len(embedded_chunks)}")

        return embedded_chunks


async def async_main():
    parser = argparse.ArgumentParser(
        description="Process PDF slides and create embeddings for RAG"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to PDF file"
    )
    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Session info (e.g., 'Nov 2024 Birmingham AI Meetup')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: slides-{pdf_name}.json)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving JSON file locally (useful when uploading directly to Supabase)"
    )

    args = parser.parse_args()

    try:
        processor = SlideProcessor()
        await processor.process(
            args.pdf,
            args.session,
            args.output,
            save_local=not args.no_save
        )
    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
