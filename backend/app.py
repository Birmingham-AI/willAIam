from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional

from services.rag_service import RAGService

app = FastAPI(
    title="willAIam Backend API",
    description="RAG API for Birmingham AI community meeting notes",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service
rag_service = RAGService()


class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class SearchResult(BaseModel):
    slide: int
    year: int
    month: int
    text: str
    score: float


class QuestionResponse(BaseModel):
    answer: str
    results: List[SearchResult]


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "willAIam Backend API",
        "version": "1.0.0"
    }


@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get a non-streaming response with supporting results.

    Request body:
    - question: The question to ask
    - top_k: Number of top results to return (default: 5)

    Returns:
    - answer: The synthesized answer
    - results: Supporting search results with similarity scores
    """
    try:
        results = rag_service.search_meeting_notes(request.question, request.top_k)
        answer = await rag_service.synthesize_answer_non_streaming(request.question, results)

        return QuestionResponse(
            answer=answer,
            results=[SearchResult(**result) for result in results]
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/ask/stream")
async def ask_question_stream(request: QuestionRequest):
    """
    Ask a question and get a streaming response.

    Request body:
    - question: The question to ask
    - top_k: Number of top results to return (default: 5)

    Returns:
    Server-Sent Events stream with the answer followed by supporting results
    """
    try:
        results = rag_service.search_meeting_notes(request.question, request.top_k)

        async def generate():
            # Stream the answer
            async for chunk in rag_service.synthesize_answer_streaming(request.question, results):
                yield f"data: {chunk}\n\n"

            # Send completion marker
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/search")
async def search_notes(question: str, top_k: int = 5):
    """
    Search meeting notes without answer synthesis.

    Query parameters:
    - question: The search query
    - top_k: Number of top results to return (default: 5)

    Returns:
    - List of search results with similarity scores
    """
    try:
        results = rag_service.search_meeting_notes(question, top_k)
        return {"results": [SearchResult(**result) for result in results]}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
