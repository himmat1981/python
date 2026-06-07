"""
routers/documents.py

Document upload and management endpoints.

Flow:
  1. POST /documents/upload
       → validate file type + size
       → extract text (PDF/DOCX/TXT/MD)
       → chunk text
       → embed chunks and store in document_chunks table
       → return document_id (UUID)

  2. Caller passes document_id to POST /chatbot/ask
       → chatbot searches both knowledge base AND the uploaded document

  3. DELETE /documents/{document_id}
       → delete all chunks for that document

  4. POST /documents/cleanup
       → delete all document chunks older than 24 hours
"""

import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File

from services.document_parser import parse_document, SUPPORTED_EXTENSIONS
from services.chunker import chunk_text
from db.vectors import store_document_chunks, delete_document, clean_expired_documents

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document and index it for vector search.

    Accepted formats: PDF, DOCX, TXT, MD (max 10 MB).

    Returns a document_id UUID. Pass it to POST /chatbot/ask
    as the `document_id` field to query against this document.

    Document chunks are automatically deleted after 24 hours.
    Use DELETE /documents/{document_id} to remove them sooner.
    """
    try:
        filename = file.filename or "document"
        ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""

        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type '{ext}'. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                ),
            )

        file_bytes = await file.read()

        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        if len(file_bytes) > MAX_FILE_SIZE:
            mb = len(file_bytes) / (1024 * 1024)
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({mb:.1f} MB). Maximum allowed size is 10 MB.",
            )

        # Extract plain text
        text = parse_document(filename, file_bytes)
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract any text from the document.",
            )

        # Chunk text (reuses the same chunker as node content)
        raw_chunks = chunk_text(text)
        if not raw_chunks:
            raise HTTPException(
                status_code=400,
                detail="Document is too short to process.",
            )

        # Build chunk dicts expected by store_document_chunks()
        document_id = str(uuid.uuid4())
        chunks = [
            {"content": chunk, "chunk_index": i}
            for i, chunk in enumerate(raw_chunks)
        ]

        # Embed and store
        await store_document_chunks(document_id, filename, chunks)

        return {
            "status":      "indexed",
            "document_id": document_id,
            "filename":    filename,
            "chunk_count": len(chunks),
            "message": (
                "Document indexed successfully. "
                "Use document_id in POST /chatbot/ask to query it. "
                "Chunks are auto-deleted after 24 hours."
            ),
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.delete("/{document_id}")
async def remove_document(document_id: str):
    """Delete an uploaded document and all its chunks from the index."""
    try:
        await delete_document(document_id)
        return {"status": "deleted", "document_id": document_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup")
async def cleanup_old_documents():
    """Delete all document chunks older than 24 hours."""
    try:
        await clean_expired_documents(max_age_hours=24)
        return {"status": "cleaned"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
