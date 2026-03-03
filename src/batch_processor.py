"""
batch_processor.py
------------------
Async batch processing pipeline for appraisal PDFs.

Processes an entire folder of PDFs concurrently using asyncio and a
thread pool executor (since pdfplumber and the OpenAI SDK are synchronous).

Architecture:
  - asyncio.Semaphore limits concurrent API calls (default: 5)
    to avoid rate-limiting from OpenAI
  - Each PDF is processed in a ThreadPoolExecutor so blocking I/O
    (PDF parsing, HTTP requests) doesn't block the event loop
  - Results are collected as they complete (asyncio.as_completed)
    so progress is shown in real time
  - Output: a CSV where each row is one document and each column
    is one extracted field (plus confidence and calibration data)

Usage (CLI):
  python -m src.batch_processor ./pdfs/ --output results.csv --concurrency 3

Usage (programmatic):
  from src.batch_processor import BatchProcessor
  processor = BatchProcessor(api_key="sk-...", concurrency=5)
  asyncio.run(processor.process_folder("./pdfs/", "results.csv"))
"""

from __future__ import annotations

import asyncio
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich import box

from .pdf_extractor import extract_pdf, ExtractionError
from .llm_extractor import LLMExtractor
from .schemas import AppraisalFields, ExtractionResponse

load_dotenv()
console = Console()


# ---------------------------------------------------------------------------
# Result data structure
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    """Result for a single PDF in the batch."""
    file_name: str
    file_path: Path
    success: bool
    response: Optional[ExtractionResponse] = None
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

# Fields to include as columns in the CSV output.
# Each field gets a _value and _confidence column.
CSV_FIELDS = [
    "property_address", "city", "state", "zip_code",
    "appraised_value", "effective_date_of_appraisal",
    "gross_living_area_sqft", "lot_size", "year_built",
    "property_type", "number_of_bedrooms", "number_of_bathrooms",
    "condition_rating", "quality_rating", "appraiser_name",
    "neighborhood_name", "market_trend",
    "comp_1_address", "comp_1_sale_price", "comp_1_gla",
    "comp_2_address", "comp_2_sale_price", "comp_2_gla",
    "comp_3_address", "comp_3_sale_price", "comp_3_gla",
]


def _build_csv_headers() -> list[str]:
    """Build the full list of CSV column headers."""
    headers = ["file_name", "processing_time_s", "status", "error"]

    # One value + confidence column per field
    for fname in CSV_FIELDS:
        headers.append(f"{fname}")
        headers.append(f"{fname}_confidence")

    # Summary columns
    headers += [
        "high_confidence_count",
        "medium_confidence_count",
        "low_confidence_count",
        "missing_count",
        "reliability_score",
        "calibration_error_count",
        "calibration_warning_count",
        "comp_quality_score",
        "comp_quality_label",
        "appraised_value_in_comp_range",
        "bracketing_satisfied",
    ]

    return headers


def _result_to_csv_row(result: BatchResult) -> dict:
    """Convert a BatchResult into a flat dict for CSV writing."""
    row: dict = {
        "file_name": result.file_name,
        "processing_time_s": round(result.processing_time_seconds, 3),
        "status": "success" if result.success else "error",
        "error": result.error_message or "",
    }

    if not result.success or result.response is None:
        # Fill all field columns with empty strings on failure
        for fname in CSV_FIELDS:
            row[fname] = ""
            row[f"{fname}_confidence"] = ""
        for col in [
            "high_confidence_count", "medium_confidence_count",
            "low_confidence_count", "missing_count",
            "reliability_score", "calibration_error_count",
            "calibration_warning_count", "comp_quality_score",
            "comp_quality_label", "appraised_value_in_comp_range",
            "bracketing_satisfied",
        ]:
            row[col] = ""
        return row

    resp = result.response
    fields: AppraisalFields = resp.fields

    # Extracted field values and confidences
    for fname in CSV_FIELDS:
        field_obj = getattr(fields, fname)
        row[fname] = field_obj.value or ""
        row[f"{fname}_confidence"] = field_obj.confidence.value

    # Confidence summary
    summary = resp.confidence_summary
    row["high_confidence_count"] = summary.get("high", 0)
    row["medium_confidence_count"] = summary.get("medium", 0)
    row["low_confidence_count"] = summary.get("low", 0)
    row["missing_count"] = summary.get("missing", 0)

    # Calibration data
    cal = resp.calibration or {}
    row["reliability_score"] = cal.get("reliability_score", "")
    row["calibration_error_count"] = cal.get("error_count", "")
    row["calibration_warning_count"] = cal.get("warning_count", "")

    # Comp analysis data
    comp = resp.comp_analysis or {}
    row["comp_quality_score"] = comp.get("comp_quality_score", "")
    row["comp_quality_label"] = comp.get("comp_quality_label", "")
    row["appraised_value_in_comp_range"] = (
        comp.get("gla_adjustment_analysis", {}).get("appraised_value_in_range", "")
    )
    bracketing = comp.get("bracketing", {})
    row["bracketing_satisfied"] = bracketing.get("is_bracketed", "") if bracketing else ""

    return row


def write_csv(results: list[BatchResult], output_path: Path) -> None:
    """Write all batch results to a CSV file."""
    headers = _build_csv_headers()
    rows = [_result_to_csv_row(r) for r in results]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Core batch processor
# ---------------------------------------------------------------------------

class BatchProcessor:
    """
    Processes a folder of appraisal PDFs concurrently and exports results to CSV.

    Args:
        api_key:     OpenAI API key.
        model:       LLM model name (default: gpt-4o).
        concurrency: Max number of PDFs to process simultaneously.
                     Higher = faster but more API rate limit risk.
                     Recommended: 3-5 for most OpenAI plans.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        concurrency: int = 5,
    ):
        self.extractor = LLMExtractor(api_key=api_key, model=model)
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)

    def _process_single_sync(self, pdf_path: Path) -> BatchResult:
        """
        Process one PDF synchronously (runs in a thread pool).
        This is the blocking work: PDF parsing + LLM API call.
        """
        start = time.time()
        try:
            pdf_result = extract_pdf(pdf_path)
            response = self.extractor.extract_fields(pdf_result)
            elapsed = time.time() - start
            return BatchResult(
                file_name=pdf_path.name,
                file_path=pdf_path,
                success=True,
                response=response,
                processing_time_seconds=elapsed,
            )
        except ExtractionError as e:
            return BatchResult(
                file_name=pdf_path.name,
                file_path=pdf_path,
                success=False,
                error_message=f"PDF extraction failed: {e}",
                processing_time_seconds=time.time() - start,
            )
        except Exception as e:
            return BatchResult(
                file_name=pdf_path.name,
                file_path=pdf_path,
                success=False,
                error_message=f"Unexpected error: {e}",
                processing_time_seconds=time.time() - start,
            )

    async def _process_single_async(
        self,
        pdf_path: Path,
        executor: ThreadPoolExecutor,
        loop: asyncio.AbstractEventLoop,
    ) -> BatchResult:
        """
        Async wrapper around _process_single_sync.
        Uses semaphore to cap concurrency, runs blocking work in thread pool.
        """
        async with self.semaphore:
            return await loop.run_in_executor(
                executor,
                self._process_single_sync,
                pdf_path,
            )

    async def process_folder(
        self,
        folder_path: Path,
        output_path: Path,
    ) -> list[BatchResult]:
        """
        Process all PDFs in a folder concurrently and write results to CSV.

        Args:
            folder_path: Directory containing PDF files.
            output_path: Path for the output CSV file.

        Returns:
            List of BatchResult objects (one per PDF).
        """
        pdf_files = sorted(folder_path.glob("*.pdf"))

        if not pdf_files:
            console.print(f"[yellow]No PDF files found in {folder_path}[/yellow]")
            return []

        console.print(
            f"\n[bold]📂 Batch processing {len(pdf_files)} PDFs[/bold] "
            f"(concurrency: {self.concurrency})\n"
        )

        results: list[BatchResult] = []
        loop = asyncio.get_event_loop()

        # Use a thread pool sized to our concurrency limit
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            tasks = [
                self._process_single_async(pdf_path, executor, loop)
                for pdf_path in pdf_files
            ]

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task("Processing PDFs...", total=len(tasks))

                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    results.append(result)
                    progress.advance(task_id)

                    status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
                    console.print(
                        f"  {status} {result.file_name} "
                        f"({result.processing_time_seconds:.1f}s)"
                    )

        # Write CSV
        write_csv(results, output_path)

        # Print summary table
        self._print_summary(results, output_path)

        return results

    def _print_summary(self, results: list[BatchResult], output_path: Path) -> None:
        """Print a summary table of batch results to the terminal."""
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        avg_time = (
            sum(r.processing_time_seconds for r in results) / len(results)
            if results else 0
        )

        table = Table(
            title="Batch Processing Summary",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Total PDFs", str(len(results)))
        table.add_row("Successful", f"[green]{len(successes)}[/green]")
        table.add_row("Failed", f"[red]{len(failures)}[/red]" if failures else "0")
        table.add_row("Avg processing time", f"{avg_time:.1f}s per PDF")
        table.add_row("Output CSV", str(output_path))

        if successes:
            avg_reliability = sum(
                (r.response.calibration or {}).get("reliability_score", 0)
                for r in successes
            ) / len(successes)
            table.add_row("Avg reliability score", f"{avg_reliability:.0%}")

        console.print()
        console.print(table)

        if failures:
            console.print("\n[red]Failed files:[/red]")
            for r in failures:
                console.print(f"  • {r.file_name}: {r.error_message}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--output", "-o",
    default="batch_results.csv",
    show_default=True,
    help="Output CSV file path.",
)
@click.option(
    "--concurrency", "-c",
    default=3,
    show_default=True,
    help="Max number of PDFs to process simultaneously.",
)
def batch(folder: str, output: str, concurrency: int):
    """
    Process all PDFs in FOLDER concurrently and export results to a CSV.

    FOLDER: Path to a directory containing appraisal PDF files.

    Example:
      python -m src.batch_processor ./sample_docs/ --output results.csv --concurrency 3
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        console.print("[red]Error: OPENAI_API_KEY not set in .env[/red]")
        raise SystemExit(1)

    model = os.getenv("MODEL_NAME", "gpt-4o")
    processor = BatchProcessor(api_key=api_key, model=model, concurrency=concurrency)

    asyncio.run(
        processor.process_folder(
            folder_path=Path(folder),
            output_path=Path(output),
        )
    )


if __name__ == "__main__":
    batch()
