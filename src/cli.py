"""
cli.py
------
Command-line interface for the Appraisal Document Parser.

Usage:
  python -m src.cli extract path/to/appraisal.pdf
  python -m src.cli extract path/to/appraisal.pdf --output results.json
  python -m src.cli compare doc1.pdf doc2.pdf
  python -m src.cli extract path/to/appraisal.pdf --show-all

The CLI is useful for:
  - Quick testing without spinning up the API server
  - Batch processing in scripts
  - Debugging extraction quality on specific documents
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

from .pdf_extractor import extract_pdf, ExtractionError
from .llm_extractor import LLMExtractor
from .schemas import FieldConfidence

load_dotenv()
console = Console()


def _get_extractor() -> LLMExtractor:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        console.print("[red]Error: OPENAI_API_KEY not set. Copy .env.example to .env and add your key.[/red]")
        sys.exit(1)
    model = os.getenv("MODEL_NAME", "gpt-4o")
    return LLMExtractor(api_key=api_key, model=model)


CONFIDENCE_COLORS = {
    FieldConfidence.HIGH: "green",
    FieldConfidence.MEDIUM: "yellow",
    FieldConfidence.LOW: "orange3",
    FieldConfidence.MISSING: "red",
}


@click.group()
def cli():
    """Appraisal Document Parser — extract structured data from appraisal PDFs."""
    pass


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Save full JSON output to this file path.")
@click.option("--show-all", is_flag=True, default=False,
              help="Show all fields including MISSING ones. Default: hide missing fields.")
@click.option("--high-only", is_flag=True, default=False,
              help="Show only HIGH confidence fields.")
def extract(pdf_path: str, output: str | None, show_all: bool, high_only: bool):
    """
    Extract structured appraisal fields from a PDF.

    PDF_PATH: Path to the appraisal PDF file.
    """
    path = Path(pdf_path)
    extractor = _get_extractor()

    console.print(f"\n[bold]📄 Extracting from:[/bold] {path.name}")

    with console.status("[bold green]Extracting PDF text..."):
        try:
            pdf_result = extract_pdf(path)
        except ExtractionError as e:
            console.print(f"[red]PDF extraction failed: {e}[/red]")
            sys.exit(1)

    console.print(
        f"  Strategy: [cyan]{pdf_result.strategy_used.value}[/cyan] | "
        f"Pages: [cyan]{pdf_result.page_count}[/cyan] | "
        f"Chars: [cyan]{pdf_result.total_chars:,}[/cyan]"
    )

    with console.status("[bold green]Running LLM extraction..."):
        result = extractor.extract_fields(pdf_result)

    console.print(f"  Processing time: [cyan]{result.processing_time_seconds}s[/cyan]\n")

    # Build results table
    table = Table(
        title=f"Extracted Fields — {path.name}",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Field", style="bold", min_width=30)
    table.add_column("Value", min_width=30)
    table.add_column("Confidence", min_width=10, justify="center")
    table.add_column("Reasoning", min_width=30, overflow="fold")

    from .schemas import AppraisalFields, ExtractedField
    for fname in AppraisalFields.model_fields:
        field_obj: ExtractedField = getattr(result.fields, fname)

        if high_only and field_obj.confidence != FieldConfidence.HIGH:
            continue
        if not show_all and field_obj.confidence == FieldConfidence.MISSING:
            continue

        color = CONFIDENCE_COLORS[field_obj.confidence]
        table.add_row(
            fname,
            field_obj.value or "—",
            f"[{color}]{field_obj.confidence.value}[/{color}]",
            field_obj.reasoning or "",
        )

    console.print(table)

    # Confidence summary
    summary = result.confidence_summary
    console.print(Panel(
        f"[green]HIGH: {summary.get('high', 0)}[/green]  "
        f"[yellow]MEDIUM: {summary.get('medium', 0)}[/yellow]  "
        f"[orange3]LOW: {summary.get('low', 0)}[/orange3]  "
        f"[red]MISSING: {summary.get('missing', 0)}[/red]",
        title="Confidence Summary",
        expand=False,
    ))

    # Save JSON if requested
    if output:
        out_path = Path(output)
        out_path.write_text(result.model_dump_json(indent=2))
        console.print(f"\n[green]✓ Full JSON saved to: {out_path}[/green]")


@cli.command()
@click.argument("pdf1_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("pdf2_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Save full JSON output to this file path.")
@click.option("--discrepancies-only", is_flag=True, default=True,
              help="Show only fields that differ between documents (default: True).")
def compare(pdf1_path: str, pdf2_path: str, output: str | None, discrepancies_only: bool):
    """
    Compare two appraisal PDFs field by field.

    PDF1_PATH: Path to the first appraisal PDF.
    PDF2_PATH: Path to the second appraisal PDF.
    """
    p1, p2 = Path(pdf1_path), Path(pdf2_path)
    extractor = _get_extractor()

    console.print(f"\n[bold]📊 Comparing:[/bold] {p1.name}  vs  {p2.name}")

    for label, path in [(p1.name, p1), (p2.name, p2)]:
        with console.status(f"Extracting {label}..."):
            try:
                pdf_result = extract_pdf(path)
                result = extractor.extract_fields(pdf_result)
                if label == p1.name:
                    result1 = result
                else:
                    result2 = result
            except (ExtractionError, Exception) as e:
                console.print(f"[red]Failed on {label}: {e}[/red]")
                sys.exit(1)

    with console.status("Generating comparison..."):
        comparison = extractor.compare_documents(result1, result2)

    # Comparison table
    table = Table(
        title="Field Comparison",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Field", style="bold", min_width=28)
    table.add_column(p1.name[:25], min_width=20)
    table.add_column(p2.name[:25], min_width=20)
    table.add_column("Match", justify="center", min_width=8)
    table.add_column("Difference", min_width=15)

    for comp in comparison.comparisons:
        if discrepancies_only and comp.are_equal:
            continue
        match_str = "[green]✓[/green]" if comp.are_equal else "[red]✗[/red]"
        table.add_row(
            comp.field_name,
            comp.doc1_value or "—",
            comp.doc2_value or "—",
            match_str,
            comp.difference_note or "",
        )

    console.print(table)

    console.print(Panel(
        f"Fields compared: [cyan]{comparison.total_fields_compared}[/cyan]  |  "
        f"Agreement: [green]{comparison.fields_in_agreement}[/green]  |  "
        f"Discrepancies: [red]{comparison.fields_with_discrepancy}[/red]  |  "
        f"Agreement rate: [bold]{comparison.agreement_rate:.1%}[/bold]",
        title="Summary Statistics",
        expand=False,
    ))

    console.print(Panel(
        comparison.summary,
        title="AI Analysis",
        border_style="blue",
        expand=False,
    ))

    if output:
        out_path = Path(output)
        out_path.write_text(comparison.model_dump_json(indent=2))
        console.print(f"\n[green]✓ Full JSON saved to: {out_path}[/green]")


if __name__ == "__main__":
    cli()
