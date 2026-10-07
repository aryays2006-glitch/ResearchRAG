from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.rag import (
    ask_research_rag,
    retriever
)

from api.indexer import process_pdf


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="ResearchRAG API",
    description="API for the ResearchRAG AI research assistant.",
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Upload directory
# --------------------------------------------------

UPLOAD_DIR = Path("data/papers")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Request model
# --------------------------------------------------

class QuestionRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask ResearchRAG."
    )


# --------------------------------------------------
# Source response model
# --------------------------------------------------

class Source(BaseModel):

    evidence_id: str
    chunk_id: str
    source: str
    page: int
    reranker_score: float


# --------------------------------------------------
# RAG response model
# --------------------------------------------------

class RAGResponse(BaseModel):

    question: str
    answer: str
    citations: list[str]
    invalid_citations: list[str]
    sources: list[Source]


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "ResearchRAG API is running."
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Upload PDF
# --------------------------------------------------

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    # Make sure the uploaded file is a PDF.
    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # Prevent unsafe filenames.
    safe_filename = Path(
        file.filename
    ).name

    pdf_path = (
        UPLOAD_DIR /
        safe_filename
    )

    try:

        # Save the uploaded PDF.
        contents = await file.read()

        with open(
            pdf_path,
            "wb"
        ) as output_file:

            output_file.write(
                contents
            )

        # Process the PDF.
        result = process_pdf(
            pdf_path
        )

        # Refresh the retriever so it
        # immediately knows about the new paper.
        retriever.refresh()

        return {
            "message": "PDF uploaded and processed successfully.",
            "filename": result["filename"],
            "pages": result["pages"],
            "chunks": result["chunks"]
        }

    except Exception as error:

        # Remove a failed upload.
        if pdf_path.exists():
            pdf_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# --------------------------------------------------
# Ask ResearchRAG
# --------------------------------------------------

@app.post(
    "/ask",
    response_model=RAGResponse
)
def ask_question(
    request: QuestionRequest
):

    result = ask_research_rag(
        request.question
    )

    return result