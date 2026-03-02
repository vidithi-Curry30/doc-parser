"""
llm_extractor.py
----------------
LLM-powered field extraction from appraisal document text.

Architecture:
  1. Build a structured system prompt defining the output JSON schema.
  2. Send truncated document text to GPT-4o in JSON mode.
  3. Parse the response back into our Pydantic AppraisalFields model.
  4. Run a post-processing pass to flag low-confidence or suspicious values
     (e.g., appraised_value of $0 or year_built in the future).

Design decisions:
  - We use JSON mode (response_format={"type": "json_object"}) to guarantee
    parseable output, avoiding markdown code fences.
  - We send the FULL field schema in the prompt so the LLM knows exactly
    what keys to return and in what shape.
  - Confidence scoring is done by the LLM for semantic fields, but we also
    run a rule-based validation pass afterward for numeric fields.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Optional

from openai import OpenAI
from pydantic import ValidationError

from .schemas import (
    AppraisalFields,
    ExtractionResponse,
    ExtractedField,
    FieldConfidence,
    ComparisonField,
    ComparisonResponse,
)
from .pdf_extractor import ExtractionResult


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert real estate document analyst specializing in residential \
property appraisal reports (URAR forms, UAD 3.6 format).

Your task: extract structured data from appraisal document text.

Return ONLY a valid JSON object. No preamble, no markdown, no explanation.

For each field, return an object with exactly these keys:
  "value"         : string or null
  "confidence"    : "high" | "medium" | "low" | "missing"
  "reasoning"     : brief string explaining how you found the value
  "source_snippet": the exact substring from the document text that contains this value, or null

Confidence guidelines:
  "high"    — you found the value clearly stated in the text
  "medium"  — you inferred it from context or it was partially visible
  "low"     — you're guessing; the text is ambiguous
  "missing" — the field is not present in the document

The JSON must have exactly these top-level keys, each mapping to the object above:
  property_address, city, state, zip_code, county, legal_description,
  assessors_parcel_number, appraised_value, effective_date_of_appraisal,
  appraisal_purpose, appraiser_name, appraiser_license_number, lender_client,
  gross_living_area_sqft, lot_size, year_built, property_type,
  number_of_bedrooms, number_of_bathrooms, number_of_stories, garage_capacity,
  basement, condition_rating, quality_rating, neighborhood_name,
  market_trend, prior_sale_price, prior_sale_date,
  comp_1_address, comp_1_sale_price, comp_1_gla,
  comp_2_address, comp_2_sale_price, comp_2_gla,
  comp_3_address, comp_3_sale_price, comp_3_gla

Numeric values like appraised_value should be returned as strings with dollar signs and commas \
preserved (e.g. "$425,000"). Dates should be kept in their original format from the document.
"""


def _build_user_message(doc_text: str, max_chars: int = 12000) -> str:
    """
    Build the user message for the extraction prompt.
    We cap text length to avoid token limit issues.
    Appraisal key fields are usually in the first ~10k chars.
    """
    truncated = doc_text[:max_chars]
    if len(doc_text) > max_chars:
        truncated += f"\n\n[... document truncated at {max_chars} chars ...]"

    return f"""Extract all appraisal fields from the following document text:

--- DOCUMENT START ---
{truncated}
--- DOCUMENT END ---
"""


# ---------------------------------------------------------------------------
# Rule-based post-validation
# ---------------------------------------------------------------------------

def _validate_numeric_field(
    field: ExtractedField, field_name: str
) -> ExtractedField:
    """
    Apply domain-specific sanity checks to numeric fields.
    Downgrades confidence if values look unrealistic.

    Examples of what this catches:
      - appraised_value of $0 or $1 (data entry error)
      - year_built of 2090 (future year — likely OCR error)
      - gross_living_area_sqft of 10 sq ft (physically impossible)
    """
    if field.value is None or field.confidence == FieldConfidence.MISSING:
        return field

    value_str = field.value.replace(",", "").replace("$", "").strip()
    try:
        numeric = float(value_str)
    except ValueError:
        return field  # Not numeric, skip

    issues = []

    if field_name == "appraised_value":
        if numeric < 1000:
            issues.append(f"Suspiciously low appraised value: {field.value}")
        elif numeric > 100_000_000:
            issues.append(f"Suspiciously high appraised value: {field.value}")

    elif field_name == "year_built":
        current_year = datetime.now().year
        if numeric < 1600 or numeric > current_year:
            issues.append(f"year_built {numeric} is out of plausible range")

    elif field_name == "gross_living_area_sqft":
        if numeric < 100:
            issues.append(f"GLA of {numeric} sqft is implausibly small")
        elif numeric > 50000:
            issues.append(f"GLA of {numeric} sqft is implausibly large for residential")

    elif field_name == "number_of_bedrooms":
        if numeric < 0 or numeric > 20:
            issues.append(f"Bedroom count {numeric} is unrealistic")

    if issues:
        updated_reasoning = field.reasoning + f" [VALIDATION WARNING: {'; '.join(issues)}]"
        # Downgrade confidence one level
        downgrade_map = {
            FieldConfidence.HIGH: FieldConfidence.MEDIUM,
            FieldConfidence.MEDIUM: FieldConfidence.LOW,
            FieldConfidence.LOW: FieldConfidence.LOW,
        }
        new_confidence = downgrade_map.get(field.confidence, field.confidence)
        return ExtractedField(
            value=field.value,
            confidence=new_confidence,
            reasoning=updated_reasoning,
            source_snippet=field.source_snippet,
        )

    return field


NUMERIC_FIELDS = {
    "appraised_value",
    "gross_living_area_sqft",
    "year_built",
    "number_of_bedrooms",
    "number_of_bathrooms",
    "prior_sale_price",
    "comp_1_sale_price",
    "comp_2_sale_price",
    "comp_3_sale_price",
}


def _post_validate(fields: AppraisalFields) -> AppraisalFields:
    """Run rule-based validation over all numeric fields."""
    data = fields.model_dump()
    for field_name in NUMERIC_FIELDS:
        if field_name in data:
            validated = _validate_numeric_field(
                ExtractedField(**data[field_name]), field_name
            )
            data[field_name] = validated.model_dump()
    return AppraisalFields(**data)


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class LLMExtractor:
    """
    Wraps the OpenAI API to provide structured field extraction
    from appraisal document text.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def extract_fields(
        self,
        extraction_result: ExtractionResult,
        max_chars: int = 12000,
    ) -> ExtractionResponse:
        """
        Run LLM extraction on a PDF ExtractionResult.

        Args:
            extraction_result: Output from pdf_extractor.extract_pdf()
            max_chars: Max document characters to send to the LLM.

        Returns:
            ExtractionResponse with structured fields and confidence data.
        """
        start_time = time.time()

        doc_text = extraction_result.get_truncated_text(max_chars)
        user_message = _build_user_message(doc_text, max_chars)

        # Call the LLM
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,   # Deterministic — we want consistent extraction
            max_tokens=4000,
        )

        raw_json_str = response.choices[0].message.content
        raw_data = json.loads(raw_json_str)

        # Parse into Pydantic model, with graceful fallback for missing keys
        try:
            fields = AppraisalFields(**raw_data)
        except ValidationError:
            # If LLM returned unexpected shape, fill missing keys with defaults
            cleaned = {}
            for fname in AppraisalFields.model_fields:
                if fname in raw_data and isinstance(raw_data[fname], dict):
                    cleaned[fname] = raw_data[fname]
                else:
                    cleaned[fname] = ExtractedField().model_dump()
            fields = AppraisalFields(**cleaned)

        # Post-validate numeric fields
        fields = _post_validate(fields)

        elapsed = time.time() - start_time

        return ExtractionResponse(
            file_name=extraction_result.file_name,
            page_count=extraction_result.page_count,
            extraction_strategy=extraction_result.strategy_used.value,
            total_chars_extracted=extraction_result.total_chars,
            fields=fields,
            confidence_summary=self._build_confidence_summary(fields),
            processing_time_seconds=round(elapsed, 3),
        )

    def _build_confidence_summary(self, fields: AppraisalFields) -> dict[str, int]:
        """Count fields per confidence level."""
        counts = {c.value: 0 for c in FieldConfidence}
        for fname in AppraisalFields.model_fields:
            f: ExtractedField = getattr(fields, fname)
            counts[f.confidence.value] += 1
        return counts

    def compare_documents(
        self,
        result1: ExtractionResponse,
        result2: ExtractionResponse,
    ) -> ComparisonResponse:
        """
        Compare two extraction results field by field.

        For numeric fields (like appraised_value), computes the dollar/unit
        difference. For text fields, checks string equality (case-insensitive,
        whitespace-normalized).

        Then calls the LLM for a narrative summary of the key discrepancies.
        """
        comparisons: list[ComparisonField] = []
        fields_in_agreement = 0
        fields_with_discrepancy = 0
        fields_only_in_doc1 = 0
        fields_only_in_doc2 = 0

        for fname in AppraisalFields.model_fields:
            f1: ExtractedField = getattr(result1.fields, fname)
            f2: ExtractedField = getattr(result2.fields, fname)

            v1 = f1.value
            v2 = f2.value

            # Determine equality
            both_missing = (
                f1.confidence == FieldConfidence.MISSING and
                f2.confidence == FieldConfidence.MISSING
            )
            only_in_1 = (
                f1.confidence != FieldConfidence.MISSING and
                f2.confidence == FieldConfidence.MISSING
            )
            only_in_2 = (
                f1.confidence == FieldConfidence.MISSING and
                f2.confidence != FieldConfidence.MISSING
            )

            if both_missing:
                are_equal = True
            elif only_in_1:
                are_equal = False
                fields_only_in_doc1 += 1
            elif only_in_2:
                are_equal = False
                fields_only_in_doc2 += 1
            else:
                # Normalize strings for comparison
                norm1 = re.sub(r"\s+", " ", (v1 or "").lower().strip())
                norm2 = re.sub(r"\s+", " ", (v2 or "").lower().strip())
                are_equal = norm1 == norm2

            if are_equal:
                fields_in_agreement += 1
            else:
                fields_with_discrepancy += 1

            # Build difference note for numeric fields
            diff_note = None
            if not are_equal and fname in NUMERIC_FIELDS and v1 and v2:
                try:
                    n1 = float(re.sub(r"[,$]", "", v1))
                    n2 = float(re.sub(r"[,$]", "", v2))
                    diff = abs(n1 - n2)
                    diff_note = f"Difference: {diff:,.2f}"
                except ValueError:
                    pass

            comparisons.append(ComparisonField(
                field_name=fname,
                doc1_value=v1,
                doc2_value=v2,
                doc1_confidence=f1.confidence,
                doc2_confidence=f2.confidence,
                are_equal=are_equal,
                difference_note=diff_note,
            ))

        total = len(comparisons)
        agreement_rate = fields_in_agreement / total if total > 0 else 0.0

        # Ask LLM for a narrative summary of the discrepancies
        summary = self._generate_comparison_summary(result1, result2, comparisons)

        return ComparisonResponse(
            doc1_name=result1.file_name,
            doc2_name=result2.file_name,
            total_fields_compared=total,
            fields_in_agreement=fields_in_agreement,
            fields_with_discrepancy=fields_with_discrepancy,
            fields_only_in_doc1=fields_only_in_doc1,
            fields_only_in_doc2=fields_only_in_doc2,
            agreement_rate=round(agreement_rate, 4),
            comparisons=comparisons,
            summary=summary,
        )

    def _generate_comparison_summary(
        self,
        result1: ExtractionResponse,
        result2: ExtractionResponse,
        comparisons: list[ComparisonField],
    ) -> str:
        """Generate a plain-English summary of the key differences between two documents."""
        discrepancies = [c for c in comparisons if not c.are_equal]
        if not discrepancies:
            return "The two documents are in full agreement on all extracted fields."

        discrepancy_text = "\n".join(
            f"  - {c.field_name}: '{c.doc1_value}' vs '{c.doc2_value}'"
            + (f" ({c.difference_note})" if c.difference_note else "")
            for c in discrepancies[:15]  # Cap at 15 to avoid huge prompt
        )

        prompt = f"""Two appraisal documents were compared. Here are the fields that differ:

{discrepancy_text}

Write a concise 3-5 sentence professional summary of these discrepancies and \
their potential significance for an appraiser reviewing both documents. \
Focus on the most impactful differences (e.g., value discrepancies, property characteristic mismatches)."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
