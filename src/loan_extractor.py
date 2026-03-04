"""
loan_extractor.py
-----------------
LLM-powered field extraction from Form 1003 (Uniform Residential Loan
Application) documents.

Architecture mirrors llm_extractor.py:
  1. Build a structured system prompt defining the Form 1003 JSON schema.
  2. Send truncated document text to GPT-4o in JSON mode.
  3. Parse the response into our Pydantic LoanFields model.
  4. Run a post-processing pass to flag suspicious values
     (e.g., credit_score outside 300-850, DTI over 100%).

Form 1003 is the standard Fannie Mae/Freddie Mac loan application used for
conventional, FHA, and VA residential mortgages.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from openai import OpenAI
from pydantic import ValidationError

from .schemas import (
    LoanFields,
    LoanExtractionResponse,
    ExtractedField,
    FieldConfidence,
)
from .pdf_extractor import ExtractionResult


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

LOAN_SYSTEM_PROMPT = """You are an expert mortgage document analyst specializing in \
residential loan applications (Form 1003 / URLA — Uniform Residential Loan Application).

Your task: extract structured data from a Form 1003 loan application document.

Return ONLY a valid JSON object. No preamble, no markdown, no explanation.

For each field, return an object with exactly these keys:
  "value"         : string or null
  "confidence"    : "high" | "medium" | "low" | "missing"
  "reasoning"     : brief string explaining how you found the value
  "source_snippet": the exact substring from the document text that contains this value, or null

Confidence guidelines:
  "high"    — value found clearly stated in the text
  "medium"  — inferred from context or partially visible
  "low"     — best guess; text is ambiguous
  "missing" — field not present in the document

The JSON must have exactly these top-level keys, each mapping to the object above:
  loan_amount, loan_purpose, loan_type, amortization_type, loan_term_months,
  interest_rate, property_address, property_type, occupancy_type,
  purchase_price, down_payment, ltv_ratio,
  borrower_name, borrower_dob, borrower_ssn_last4, borrower_phone, borrower_email,
  co_borrower_name, employer_name, employment_years,
  gross_monthly_income, base_monthly_income,
  checking_savings_balance, monthly_debt_payments, dti_ratio,
  credit_score, lender_name, loan_officer_name, application_date

Field-specific guidance:
  - loan_amount, purchase_price, down_payment: return with $ and commas (e.g. "$450,000")
  - interest_rate, ltv_ratio, dti_ratio: return as percentage string (e.g. "6.75%")
  - loan_term_months: return as integer string (e.g. "360" for 30-year)
  - borrower_ssn_last4: return ONLY the last 4 digits if visible, never the full SSN
  - credit_score: return as plain integer string (e.g. "742")
  - Dates should be kept in their original format from the document
"""


def _build_loan_user_message(doc_text: str, max_chars: int = 12000) -> str:
    truncated = doc_text[:max_chars]
    if len(doc_text) > max_chars:
        truncated += f"\n\n[... document truncated at {max_chars} chars ...]"

    return f"""Extract all Form 1003 loan application fields from the following document text:

--- DOCUMENT START ---
{truncated}
--- DOCUMENT END ---
"""


# ---------------------------------------------------------------------------
# Rule-based post-validation
# ---------------------------------------------------------------------------

def _validate_loan_field(field: ExtractedField, field_name: str) -> ExtractedField:
    """
    Apply domain-specific sanity checks to loan application numeric fields.
    Downgrades confidence if values look unrealistic.
    """
    if field.value is None or field.confidence == FieldConfidence.MISSING:
        return field

    value_str = field.value.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        numeric = float(value_str)
    except ValueError:
        return field

    issues = []

    if field_name == "loan_amount":
        if numeric < 10_000:
            issues.append(f"Loan amount {field.value} is suspiciously low")
        elif numeric > 50_000_000:
            issues.append(f"Loan amount {field.value} exceeds residential limits")

    elif field_name == "credit_score":
        if numeric < 300 or numeric > 850:
            issues.append(f"Credit score {numeric} is outside valid range (300-850)")

    elif field_name == "dti_ratio":
        if numeric < 0 or numeric > 100:
            issues.append(f"DTI ratio {numeric}% is outside valid range (0-100%)")
        elif numeric > 65:
            issues.append(f"DTI ratio {numeric}% is very high — likely approval risk")

    elif field_name == "ltv_ratio":
        if numeric < 0 or numeric > 150:
            issues.append(f"LTV ratio {numeric}% is outside plausible range")

    elif field_name == "interest_rate":
        if numeric < 0.5 or numeric > 30:
            issues.append(f"Interest rate {numeric}% is outside plausible range")

    elif field_name == "loan_term_months":
        valid_terms = {60, 120, 180, 240, 360, 480}
        if numeric not in valid_terms:
            issues.append(f"Loan term {numeric} months is unusual (expected 60/120/180/240/360)")

    if issues:
        updated_reasoning = field.reasoning + f" [VALIDATION WARNING: {'; '.join(issues)}]"
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


LOAN_NUMERIC_FIELDS = {
    "loan_amount",
    "purchase_price",
    "down_payment",
    "gross_monthly_income",
    "base_monthly_income",
    "checking_savings_balance",
    "monthly_debt_payments",
    "interest_rate",
    "ltv_ratio",
    "dti_ratio",
    "credit_score",
    "loan_term_months",
}


def _post_validate_loan(fields: LoanFields) -> LoanFields:
    """Run rule-based validation over all loan numeric fields."""
    data = fields.model_dump()
    for field_name in LOAN_NUMERIC_FIELDS:
        if field_name in data:
            validated = _validate_loan_field(
                ExtractedField(**data[field_name]), field_name
            )
            data[field_name] = validated.model_dump()
    return LoanFields(**data)


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class LoanExtractor:
    """
    Wraps the OpenAI API to provide structured field extraction
    from Form 1003 mortgage loan application documents.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def extract_fields(
        self,
        extraction_result: ExtractionResult,
        max_chars: int = 12000,
    ) -> LoanExtractionResponse:
        """
        Run LLM extraction on a PDF ExtractionResult containing a Form 1003.

        Args:
            extraction_result: Output from pdf_extractor.extract_pdf()
            max_chars: Max document characters to send to the LLM.

        Returns:
            LoanExtractionResponse with structured loan fields and confidence data.
        """
        start_time = time.time()

        doc_text = extraction_result.get_truncated_text(max_chars)
        user_message = _build_loan_user_message(doc_text, max_chars)

        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": LOAN_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=4000,
        )

        raw_json_str = response.choices[0].message.content
        raw_data = json.loads(raw_json_str)

        try:
            fields = LoanFields(**raw_data)
        except ValidationError:
            cleaned = {}
            for fname in LoanFields.model_fields:
                if fname in raw_data and isinstance(raw_data[fname], dict):
                    cleaned[fname] = raw_data[fname]
                else:
                    cleaned[fname] = ExtractedField().model_dump()
            fields = LoanFields(**cleaned)

        fields = _post_validate_loan(fields)

        elapsed = time.time() - start_time

        return LoanExtractionResponse(
            file_name=extraction_result.file_name,
            page_count=extraction_result.page_count,
            extraction_strategy=extraction_result.strategy_used.value,
            total_chars_extracted=extraction_result.total_chars,
            fields=fields,
            confidence_summary=self._build_confidence_summary(fields),
            processing_time_seconds=round(elapsed, 3),
        )

    def _build_confidence_summary(self, fields: LoanFields) -> dict[str, int]:
        """Count fields per confidence level."""
        counts = {c.value: 0 for c in FieldConfidence}
        for fname in LoanFields.model_fields:
            f: ExtractedField = getattr(fields, fname)
            counts[f.confidence.value] += 1
        return counts
