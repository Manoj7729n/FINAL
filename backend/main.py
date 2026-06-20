```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
import uvicorn

from services.document_processor import DocumentProcessor
from services.vector_store import VectorStore
from services.llm_service import LLMService

app = FastAPI(
    title="RAG Knowledge Assistant",
    version="1.0.0"
)

# CORS Configuration
frontend_origins = (
    os.getenv("FRONTEND_URLS")
    or os.getenv("FRONTEND_URL")
    or "http://localhost:3000,http://localhost:5173"
)

if isinstance(frontend_origins, str):
    origins_list = [
        o.strip()
        for o in frontend_origins.split(",")
        if o.strip()
    ]
else:
    origins_list = list(frontend_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded services
doc_processor = None
vector_store = None
llm_service = None


def get_doc_processor():
    global doc_processor

    if doc_processor is None:
        doc_processor = DocumentProcessor()

    return doc_processor


def get_vector_store():
    global vector_store

    if vector_store is None:
        vector_store = VectorStore()

    return vector_store


def get_llm_service():
    global llm_service

    if llm_service is None:
        llm_service = LLMService()

    return llm_service


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    query: str


@app.get("/")
async def root():
    return {
        "message": "RAG Knowledge Assistant API",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "documents_indexed": get_vector_store().get_document_count()
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed_extensions = [".pdf", ".txt", ".md", ".docx"]

    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    content = await file.read()

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB."
        )

    try:
        chunks = get_doc_processor().process_file(
            content,
            file.filename,
            file_ext
        )

        doc_ids = get_vector_store().add_documents(
            chunks,
            file.filename
        )

        return JSONResponse(
            content={
                "message": f"Successfully processed '{file.filename}'",
                "filename": file.filename,
                "chunks_created": len(chunks),
                "document_ids": doc_ids,
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    if get_vector_store().get_document_count() == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents indexed. Please upload documents first."
        )

    try:
        relevant_chunks = get_vector_store().search(
            request.query,
            top_k=request.top_k
        )

        if not relevant_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant context found for your query."
            )

        answer = get_llm_service().generate_answer(
            request.query,
            relevant_chunks
        )

        sources = []
        seen = set()

        for chunk in relevant_chunks:
            source_key = (
                f"{chunk['filename']}_{chunk.get('page', 0)}"
            )

            if source_key not in seen:
                seen.add(source_key)

                sources.append(
                    {
                        "filename": chunk["filename"],
                        "page": chunk.get("page"),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "relevance_score": round(
                            chunk.get("score", 0),
                            4
                        ),
                        "excerpt": (
                            chunk["text"][:200] + "..."
                            if len(chunk["text"]) > 200
                            else chunk["text"]
                        ),
                    }
                )

        return QueryResponse(
            answer=answer,
            sources=sources,
            query=request.query
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/documents")
async def list_documents():
    docs = get_vector_store().list_documents()

    return {
        "documents": docs,
        "total": len(docs)
    }


@app.delete("/documents/{filename}")
async def delete_document(filename: str):

    deleted = get_vector_store().delete_document(filename)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found."
        )

    return {
        "message": f"Document '{filename}' deleted successfully."
    }


@app.delete("/documents")
async def clear_all_documents():
    get_vector_store().clear_all()

    return {
        "message": "All documents cleared successfully."
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))

    reload = (
        os.getenv("RELOAD", "false").lower()
        in ("1", "true", "yes")
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=reload
    )
```
