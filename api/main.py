import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging

from rag.retriever import RAGRetriever
from rag.prompt_builder import build_prompt
from rag.llm import generate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API", version="1.0.0")

# CORS configuration
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG retriever
retriever: Optional[RAGRetriever] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]


class HealthResponse(BaseModel):
    status: str


class IngestResponse(BaseModel):
    chunks_indexed: int


@app.on_event("startup")
async def startup_event():
    global retriever
    try:
        retriever = RAGRetriever()
        if retriever.is_index_loaded():
            logger.info(f"Index loaded, {retriever.get_chunk_count()} chunks ready")
        else:
            logger.warning("No index found. POST /api/ingest first.")
    except Exception as e:
        logger.error(f"Error initializing retriever: {e}")
        retriever = RAGRetriever()


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="ok")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with RAG pipeline"""
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized")

    if not retriever.is_index_loaded():
        raise HTTPException(
            status_code=503,
            detail="Index not found. Run POST /api/ingest first.",
        )

    try:
        # Retrieve context from documents
        retrieved_chunks = retriever.retrieve(request.question, top_k=5)

        # Build prompt with context and history
        prompt = build_prompt(request.question, retrieved_chunks, request.history)

        # Generate answer using LLM
        answer = generate_answer(prompt)

        # Extract page sources from retrieved chunks
        sources = []
        for chunk in retrieved_chunks:
            if "page" in chunk:
                page_num = chunk["page"]
                if f"Page {page_num}" not in sources:
                    sources.append(f"Page {page_num}")

        return ChatResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest():
    """Ingest PDF and create FAISS index"""
    global retriever
    try:
        from rag.ingest import ingest_pdf

        chunks_count = ingest_pdf()

        # Reload retriever
        retriever = RAGRetriever()

        logger.info(f"Ingestion complete: {chunks_count} chunks indexed")
        return IngestResponse(chunks_indexed=chunks_count)

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"PDF file not found: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error during ingest: {e}")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7071)
