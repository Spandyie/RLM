"""
RLM Document Chatbot - FastAPI Backend

Based on "Recursive Language Models" by Zhang, Kraska, and Khattab (2025)
https://arxiv.org/abs/2512.24601

API Endpoints:
- POST /upload    - Upload a document
- POST /chat      - Chat with documents using RLM
- GET  /documents - List uploaded documents
- GET  /health    - Health check
"""

from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

from .config import settings
from .llm import OllamaClient
from .rlm import RLMEngine, RecursiveSummarizer
from .documents import DocumentProcessor, DocumentStore


# Global instances
llm_client: Optional[OllamaClient] = None
rlm_engine: Optional[RLMEngine] = None
summarizer: Optional[RecursiveSummarizer] = None
doc_processor: Optional[DocumentProcessor] = None
doc_store: Optional[DocumentStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup, cleanup on shutdown."""
    global llm_client, rlm_engine, summarizer, doc_processor, doc_store

    llm_client = OllamaClient(
        base_url=settings.ollama.base_url,
        model=settings.ollama.model,
        timeout=settings.ollama.timeout,
    )

    rlm_engine = RLMEngine(
        llm_client=llm_client,
        max_iterations=settings.rlm.max_iterations,
    )

    summarizer = RecursiveSummarizer(
        llm_client=llm_client,
        chunk_size=settings.document.chunk_size,
    )

    doc_processor = DocumentProcessor(
        chunk_size=settings.document.chunk_size,
        chunk_overlap=settings.document.chunk_overlap,
    )

    doc_store = DocumentStore(
        persist_directory=settings.chroma.persist_directory,
        collection_name=settings.chroma.collection_name,
    )

    print("RLM Chatbot ready!")
    print(f"Model: {settings.ollama.model}")

    yield

    print("Shutting down...")


app = FastAPI(
    title="RLM Document Chatbot",
    description="Recursive Language Model chatbot (Zhang et al. 2025)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ChatRequest(BaseModel):
    query: str
    doc_id: Optional[str] = None
    use_retrieval: bool = True


class ChatResponse(BaseModel):
    query: str
    response: str
    context_length: int
    llm_calls: int
    steps: List[dict]
    success: bool


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    model_available: bool
    documents_count: int


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health."""
    ollama_ok = await llm_client.check_health() if llm_client else False
    model_ok = await llm_client.check_model_available() if llm_client else False
    stats = doc_store.get_stats() if doc_store else {"total_documents": 0}

    return HealthResponse(
        status="healthy" if ollama_ok and model_ok else "degraded",
        ollama_connected=ollama_ok,
        model_available=model_ok,
        documents_count=stats["total_documents"],
    )


@app.post("/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document."""
    if not doc_processor or not doc_store:
        raise HTTPException(503, "Not ready")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    try:
        processed = await doc_processor.process(
            file_content=content,
            filename=file.filename or "unknown",
        )
    except Exception as e:
        raise HTTPException(400, f"Error: {e}")

    doc_store.add_document(processed)

    return DocumentInfo(
        doc_id=processed.doc_id,
        filename=processed.filename,
        chunk_count=len(processed.chunks),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with documents using RLM."""
    if not rlm_engine or not doc_store:
        raise HTTPException(503, "Not ready")

    # Build context (retrieved chunks or full document)
    if request.use_retrieval:
        results = doc_store.search(
            query=request.query,
            n_results=settings.document.max_chunks_per_query,
            doc_id=request.doc_id,
        )

        if not results:
            context = "No documents uploaded. Please upload a document first."
        else:
            parts = []
            for i, r in enumerate(results, 1):
                filename = r.chunk.metadata.get("filename", "Unknown")
                parts.append(f"[Source {i} - {filename}]\n{r.chunk.text}")
            context = "\n\n---\n\n".join(parts)
    else:
        if request.doc_id:
            chunks = doc_store.get_document_chunks(request.doc_id)
        else:
            chunks = doc_store.get_all_chunks()

        if not chunks:
            context = "No documents uploaded. Please upload a document first."
        else:
            parts = []
            for i, c in enumerate(chunks, 1):
                filename = c.metadata.get("filename", "Unknown")
                parts.append(f"[Chunk {i} - {filename}]\n{c.text}")
            context = "\n\n---\n\n".join(parts)

    # Run RLM
    result = await rlm_engine.run(request.query, context)

    return ChatResponse(
        query=result.query,
        response=result.final_answer,
        context_length=result.context_length,
        llm_calls=result.total_llm_calls,
        steps=[s.to_dict() for s in result.steps],
        success=result.success,
    )


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List uploaded documents."""
    if not doc_store:
        raise HTTPException(503, "Not ready")

    docs = doc_store.list_documents()
    return [
        DocumentInfo(
            doc_id=d["doc_id"],
            filename=d["filename"],
            chunk_count=d["chunk_count"],
        )
        for d in docs
    ]


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document."""
    if not doc_store:
        raise HTTPException(503, "Not ready")

    count = doc_store.delete_document(doc_id)
    if count == 0:
        raise HTTPException(404, "Not found")

    return {"status": "deleted", "chunks_removed": count}


@app.get("/documents/{doc_id}/summary")
async def get_summary(doc_id: str):
    """Get document summary."""
    if not doc_store or not summarizer:
        raise HTTPException(503, "Not ready")

    chunks = doc_store.get_document_chunks(doc_id)
    if not chunks:
        raise HTTPException(404, "Not found")

    full_text = "\n\n".join(c.text for c in chunks)
    result = await summarizer.summarize(full_text)

    return {
        "doc_id": doc_id,
        "summary": result.summary,
        "chunk_count": result.chunk_count,
        "levels": result.levels,
    }
