"""
schemas.py
----------
Pydantic models that define the shape of our extraction outputs.

Why Pydantic?
  - Automatic validation: if the LLM returns a string where we expect a float,
    Pydantic will either coerce it or raise a clear error.
  - JSON serialization built-in (needed for API responses).
  - Self-documenting: the models double as an API schema.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class FieldConfidence(str, Enum):
    """
    How confident are we in an extracted field value?

    HIGH   — value found directly in text, no ambiguity
    MEDIUM — value inferred or partially matched
    LOW    — best guess; LLM uncertain or field not clearly present
    MISSING — field not found at all
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MISSING = "missing"


class ExtractedField(BaseModel):
    """A single extracted field with its value and confidence score."""
    value: Optional[str] = Field(
        None,
        description="The extracted value as a string, or null if not found."
    )
    confidence: FieldConfidence = Field(
        FieldConfidence.MISSING,
        description="Confidence level of this extraction."
    )
    reasoning: str = Field(
        "",
        description="Brief explanation of how the value was determined."
    )
    source_snippet: Optional[str] = Field(
        None,
        description="The exact text snippet from the document this was pulled from."
    )


class AppraisalFields(BaseModel):
    """
    Structured schema for residential property appraisal fields.

    These fields are drawn from standard UAD (Uniform Appraisal Dataset)
    forms used by Fannie Mae and Freddie Mac.
    """

    # --- Property Identification ---
    property_address: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Full street address of the subject property."
    )
    city: ExtractedField = Field(default_factory=ExtractedField)
    state: ExtractedField = Field(default_factory=ExtractedField)
    zip_code: ExtractedField = Field(default_factory=ExtractedField)
    county: ExtractedField = Field(default_factory=ExtractedField)
    legal_description: ExtractedField = Field(default_factory=ExtractedField)
    assessors_parcel_number: ExtractedField = Field(
        default_factory=ExtractedField,
        description="APN or Tax ID for the property."
    )

    # --- Appraisal Details ---
    appraised_value: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Final appraised market value in USD."
    )
    effective_date_of_appraisal: ExtractedField = Field(default_factory=ExtractedField)
    appraisal_purpose: ExtractedField = Field(
        default_factory=ExtractedField,
        description="E.g. purchase, refinance, estate."
    )
    appraiser_name: ExtractedField = Field(default_factory=ExtractedField)
    appraiser_license_number: ExtractedField = Field(default_factory=ExtractedField)
    lender_client: ExtractedField = Field(default_factory=ExtractedField)

    # --- Property Characteristics ---
    gross_living_area_sqft: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Above-grade living area in square feet (GLA)."
    )
    lot_size: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Lot area (acres or sq ft)."
    )
    year_built: ExtractedField = Field(default_factory=ExtractedField)
    property_type: ExtractedField = Field(
        default_factory=ExtractedField,
        description="E.g. single-family, condo, multi-family."
    )
    number_of_bedrooms: ExtractedField = Field(default_factory=ExtractedField)
    number_of_bathrooms: ExtractedField = Field(default_factory=ExtractedField)
    number_of_stories: ExtractedField = Field(default_factory=ExtractedField)
    garage_capacity: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Number of cars the garage holds."
    )
    basement: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Basement presence and sq footage."
    )
    condition_rating: ExtractedField = Field(
        default_factory=ExtractedField,
        description="UAD condition rating (C1-C6)."
    )
    quality_rating: ExtractedField = Field(
        default_factory=ExtractedField,
        description="UAD quality rating (Q1-Q6)."
    )

    # --- Market / Neighborhood ---
    neighborhood_name: ExtractedField = Field(default_factory=ExtractedField)
    market_trend: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Increasing, stable, or declining."
    )
    prior_sale_price: ExtractedField = Field(default_factory=ExtractedField)
    prior_sale_date: ExtractedField = Field(default_factory=ExtractedField)

    # --- Comparable Sales ---
    comp_1_address: ExtractedField = Field(default_factory=ExtractedField)
    comp_1_sale_price: ExtractedField = Field(default_factory=ExtractedField)
    comp_1_gla: ExtractedField = Field(default_factory=ExtractedField)
    comp_2_address: ExtractedField = Field(default_factory=ExtractedField)
    comp_2_sale_price: ExtractedField = Field(default_factory=ExtractedField)
    comp_2_gla: ExtractedField = Field(default_factory=ExtractedField)
    comp_3_address: ExtractedField = Field(default_factory=ExtractedField)
    comp_3_sale_price: ExtractedField = Field(default_factory=ExtractedField)
    comp_3_gla: ExtractedField = Field(default_factory=ExtractedField)

    def confidence_summary(self) -> dict[str, int]:
        """
        Count how many fields fall into each confidence bucket.
        Useful for a quick quality check on the extraction.
        """
        counts = {c.value: 0 for c in FieldConfidence}
        for field_name in AppraisalFields.model_fields:
            f: ExtractedField = getattr(self, field_name)
            counts[f.confidence.value] += 1
        return counts

    def high_confidence_fields(self) -> dict[str, str]:
        """Return only fields with HIGH confidence as a flat dict."""
        result = {}
        for fname in AppraisalFields.model_fields:
            f: ExtractedField = getattr(self, fname)
            if f.confidence == FieldConfidence.HIGH and f.value:
                result[fname] = f.value
        return result

    def to_flat_dict(self) -> dict[str, Optional[str]]:
        """Flatten to field_name -> value for easy display/export."""
        return {
            fname: getattr(self, fname).value
            for fname in AppraisalFields.model_fields
        }


class ExtractionResponse(BaseModel):
    """Top-level API response for a single document extraction."""
    file_name: str
    page_count: int
    extraction_strategy: str
    total_chars_extracted: int
    fields: AppraisalFields
    confidence_summary: dict[str, int]
    processing_time_seconds: float
    calibration: Optional[dict] = None
    comp_analysis: Optional[dict] = None


class ComparisonField(BaseModel):
    """Comparison of a single field across two documents."""
    field_name: str
    doc1_value: Optional[str]
    doc2_value: Optional[str]
    doc1_confidence: FieldConfidence
    doc2_confidence: FieldConfidence
    are_equal: bool
    difference_note: Optional[str] = None  # e.g. "values differ by $50,000"


class ComparisonResponse(BaseModel):
    """Response for comparing two appraisal documents side-by-side."""
    doc1_name: str
    doc2_name: str
    total_fields_compared: int
    fields_in_agreement: int
    fields_with_discrepancy: int
    fields_only_in_doc1: int
    fields_only_in_doc2: int
    agreement_rate: float   # 0.0 to 1.0
    comparisons: list[ComparisonField]
    summary: str            # LLM-generated narrative summary of differences
