"""
api.py
------
FastAPI REST API for the Appraisal Document Parser.

Endpoints:
  POST /extract          — upload a PDF, get structured fields back
  POST /compare          — upload two PDFs, get a field-by-field comparison
  GET  /health           — liveness check
  GET  /schema           — returns the list of all fields we extract

Design:
  - All endpoints are async for scalability.
  - Files are validated (must be PDFs) before processing.
  - Errors are surfaced with meaningful HTTP status codes and messages.
  - Processing time is included in every response for performance visibility.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from .pdf_extractor import ExtractionError, extract_pdf
from .llm_extractor import LLMExtractor
from .schemas import AppraisalFields, ExtractionResponse, ComparisonResponse

load_dotenv()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

extractor: LLMExtractor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: initialize the LLM extractor once at startup
    rather than on every request. This avoids re-creating the OpenAI client
    repeatedly and allows us to fail fast if the API key is missing.
    """
    global extractor
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise RuntimeError(
            "OPENAI_API_KEY environment variable not set. "
            "Copy .env.example to .env and add your key."
        )
    model = os.getenv("MODEL_NAME", "gpt-4o")
    extractor = LLMExtractor(api_key=api_key, model=model)
    yield
    extractor = None


app = FastAPI(
    title="Appraisal Document Parser API",
    description=(
        "AI-powered extraction of structured fields from residential property "
        "appraisal reports (URAR/UAD 3.6 format). Supports single-document "
        "extraction and two-document comparison with confidence scoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_pdf_upload(file: UploadFile) -> None:
    """Raise a 400 if the uploaded file isn't a PDF."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{file.filename}' is not a PDF. Only .pdf files are accepted.",
        )
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        # content_type check is lenient — some clients send octet-stream
        pass  # We'll rely on pdfplumber to fail loudly if it's not a PDF


async def _read_and_extract(file: UploadFile) -> ExtractionResponse:
    """Read an uploaded PDF and run both extraction stages."""
    _validate_pdf_upload(file)
    file_bytes = await file.read()

    try:
        extraction_result = extract_pdf(file_bytes, file_name=file.filename)
    except ExtractionError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not extract text from PDF: {e}",
        )

    return extractor.extract_fields(extraction_result)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
async def health_check():
    """Liveness probe. Returns 200 if the service is up and the LLM client is initialized."""
    return {
        "status": "ok",
        "llm_ready": extractor is not None,
        "model": os.getenv("MODEL_NAME", "gpt-4o"),
    }


@app.get("/schema", tags=["Meta"])
async def get_schema():
    """
    Returns the list of all fields this API can extract, with their descriptions.
    Useful for clients that want to know what to expect in an /extract response.
    """
    field_info = {}
    for fname, fmeta in AppraisalFields.model_fields.items():
        field_info[fname] = {
            "description": fmeta.description or "",
        }
    return {"fields": field_info, "total_fields": len(field_info)}


@app.post(
    "/extract",
    response_model=ExtractionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Extraction"],
    summary="Extract structured fields from a single appraisal PDF",
)
async def extract_document(
    file: UploadFile = File(
        ...,
        description="A PDF appraisal report (URAR/UAD 3.6 format recommended)."
    ),
):
    """
    Upload a PDF appraisal report and receive structured field extraction.

    The response includes:
    - All ~35 standard appraisal fields (address, value, property characteristics, comps, etc.)
    - Per-field confidence score (high / medium / low / missing)
    - Per-field reasoning and source snippet from the document
    - Confidence summary (count per confidence level)
    - Processing metadata (time, page count, extraction strategy used)

    The extraction pipeline:
    1. PDF text extraction (pdfplumber with PyMuPDF fallback)
    2. LLM structured extraction (GPT-4o in JSON mode)
    3. Rule-based post-validation for numeric fields
    """
    return await _read_and_extract(file)


@app.post(
    "/compare",
    response_model=ComparisonResponse,
    status_code=status.HTTP_200_OK,
    tags=["Comparison"],
    summary="Compare two appraisal PDFs field by field",
)
async def compare_documents(
    file1: UploadFile = File(..., description="First appraisal PDF."),
    file2: UploadFile = File(..., description="Second appraisal PDF."),
):
    """
    Upload two PDF appraisal reports and receive a detailed field-by-field comparison.

    The response includes:
    - Agreement rate (what % of fields match between documents)
    - Field-by-field comparison with both values and confidence levels
    - Numeric difference notes for value fields (e.g., "Difference: $15,000")
    - LLM-generated narrative summary of key discrepancies

    Use cases:
    - Reconciling two appraisals for the same property
    - Detecting data entry inconsistencies
    - Compliance review: comparing a draft report vs. a final report
    """
    # Extract both documents (these could be parallelized with asyncio.gather
    # in a production setting, but we keep it sequential for clarity)
    result1 = await _read_and_extract(file1)
    result2 = await _read_and_extract(file2)

    return extractor.compare_documents(result1, result2)


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Catch-all for unexpected errors — return 500 with a message instead of crashing."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
        },
    )
