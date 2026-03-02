"""
pdf_extractor.py
----------------
Multi-strategy PDF text extraction pipeline.

Strategy hierarchy (tried in order):
  1. pdfplumber  — best for text-layer PDFs with table awareness
  2. PyMuPDF     — faster fallback, handles more PDF variants
  3. Error       — raises ExtractionError if both fail

For each strategy, we extract:
  - raw text (full document)
  - page-level chunks
  - detected tables (as lists of rows)

Why two libraries?
  pdfplumber is more accurate on structured forms (which appraisal docs are),
  but fails on some encrypted or oddly-encoded PDFs where PyMuPDF succeeds.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import pdfplumber
import fitz  # PyMuPDF


class ExtractionStrategy(str, Enum):
    PDFPLUMBER = "pdfplumber"
    PYMUPDF = "pymupdf"


class ExtractionError(Exception):
    pass


@dataclass
class PageData:
    """Text and table data for a single PDF page."""
    page_number: int          # 1-indexed
    raw_text: str
    tables: list[list[list[str]]]   # tables[i][row][col]
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.raw_text)


@dataclass
class ExtractionResult:
    """Full extraction result from a PDF document."""
    file_name: str
    strategy_used: ExtractionStrategy
    pages: list[PageData]
    full_text: str = field(init=False)
    total_chars: int = field(init=False)
    page_count: int = field(init=False)

    def __post_init__(self):
        self.full_text = "\n\n--- PAGE BREAK ---\n\n".join(
            p.raw_text for p in self.pages
        )
        self.total_chars = sum(p.char_count for p in self.pages)
        self.page_count = len(self.pages)

    def get_truncated_text(self, max_chars: int = 12000) -> str:
        """
        Returns text truncated to max_chars.
        We truncate from the END because appraisal summaries
        and key fields tend to be near the top of reports.
        """
        return self.full_text[:max_chars]


def _extract_with_pdfplumber(file_bytes: bytes, file_name: str) -> ExtractionResult:
    """
    Primary extraction strategy.
    pdfplumber is particularly good at:
      - Handling multi-column layouts
      - Detecting table boundaries using whitespace analysis
      - Preserving reading order
    """
    pages: list[PageData] = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            raw_text = page.extract_text() or ""

            # Extract tables — pdfplumber returns None if no tables found
            raw_tables = page.extract_tables() or []
            # Normalize: replace None cells with empty string
            cleaned_tables = [
                [
                    [cell if cell is not None else "" for cell in row]
                    for row in table
                ]
                for table in raw_tables
            ]

            pages.append(PageData(
                page_number=i + 1,
                raw_text=raw_text,
                tables=cleaned_tables,
            ))

    return ExtractionResult(
        file_name=file_name,
        strategy_used=ExtractionStrategy.PDFPLUMBER,
        pages=pages,
    )


def _extract_with_pymupdf(file_bytes: bytes, file_name: str) -> ExtractionResult:
    """
    Fallback extraction strategy.
    PyMuPDF (fitz) is faster and handles more edge cases,
    but has weaker table detection.
    """
    pages: list[PageData] = []

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        raw_text = page.get_text("text") or ""
        pages.append(PageData(
            page_number=i + 1,
            raw_text=raw_text,
            tables=[],  # PyMuPDF table extraction requires extra config
        ))
    doc.close()

    return ExtractionResult(
        file_name=file_name,
        strategy_used=ExtractionStrategy.PYMUPDF,
        pages=pages,
    )


def extract_pdf(
    file_input: bytes | str | Path,
    file_name: Optional[str] = None,
    prefer_strategy: Optional[ExtractionStrategy] = None,
) -> ExtractionResult:
    """
    Main entry point for PDF extraction.

    Args:
        file_input: PDF as raw bytes, or a file path (str or Path).
        file_name:  Optional display name. Auto-detected from path if not given.
        prefer_strategy: Force a specific strategy (default: auto).

    Returns:
        ExtractionResult with full text and per-page data.

    Raises:
        ExtractionError: If all strategies fail.
    """
    # Normalize input to bytes
    if isinstance(file_input, (str, Path)):
        path = Path(file_input)
        file_name = file_name or path.name
        file_bytes = path.read_bytes()
    else:
        file_bytes = file_input
        file_name = file_name or "unknown.pdf"

    strategies = {
        ExtractionStrategy.PDFPLUMBER: _extract_with_pdfplumber,
        ExtractionStrategy.PYMUPDF: _extract_with_pymupdf,
    }

    # Build ordered list of strategies to try
    if prefer_strategy:
        order = [prefer_strategy] + [s for s in strategies if s != prefer_strategy]
    else:
        order = list(strategies.keys())  # default: pdfplumber first

    last_error: Optional[Exception] = None
    for strategy in order:
        try:
            result = strategies[strategy](file_bytes, file_name)
            # Reject results with suspiciously little text (likely extraction failure)
            if result.total_chars < 50 and strategy != order[-1]:
                raise ExtractionError(
                    f"{strategy} returned only {result.total_chars} chars — trying next strategy"
                )
            return result
        except ExtractionError:
            raise
        except Exception as e:
            last_error = e
            continue

    raise ExtractionError(
        f"All PDF extraction strategies failed. Last error: {last_error}"
    )
