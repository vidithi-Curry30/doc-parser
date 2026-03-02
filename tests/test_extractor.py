"""
tests/test_extractor.py
-----------------------
Unit tests for the extraction pipeline.

These tests do NOT call the real OpenAI API — they mock it.
This means the tests run fast and work without an API key.

To run:
  pytest tests/ -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.pdf_extractor import ExtractionResult, ExtractionStrategy, PageData
from src.schemas import (
    AppraisalFields,
    ExtractedField,
    FieldConfidence,
    ExtractionResponse,
)
from src.llm_extractor import LLMExtractor, _validate_numeric_field, _post_validate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_extraction_result(text: str = "Sample appraisal text.") -> ExtractionResult:
    """Build a minimal ExtractionResult for testing."""
    page = PageData(page_number=1, raw_text=text, tables=[])
    return ExtractionResult(
        file_name="test.pdf",
        strategy_used=ExtractionStrategy.PDFPLUMBER,
        pages=[page],
    )


def make_mock_fields() -> dict:
    """Return a minimal valid LLM response dict with all required keys."""
    base_field = {"value": None, "confidence": "missing", "reasoning": "", "source_snippet": None}
    fields = {fname: base_field.copy() for fname in AppraisalFields.model_fields}
    # Override a few with real values
    fields["appraised_value"] = {
        "value": "$425,000",
        "confidence": "high",
        "reasoning": "Found on page 1: 'Appraised Value: $425,000'",
        "source_snippet": "Appraised Value: $425,000",
    }
    fields["property_address"] = {
        "value": "123 Main St",
        "confidence": "high",
        "reasoning": "Address clearly stated at top of form.",
        "source_snippet": "Subject Property Address: 123 Main St",
    }
    fields["year_built"] = {
        "value": "1995",
        "confidence": "high",
        "reasoning": "Year built found in property description.",
        "source_snippet": "Year Built: 1995",
    }
    return fields


# ---------------------------------------------------------------------------
# pdf_extractor tests
# ---------------------------------------------------------------------------

class TestExtractionResult:
    def test_full_text_joins_pages(self):
        pages = [
            PageData(page_number=1, raw_text="Page one text", tables=[]),
            PageData(page_number=2, raw_text="Page two text", tables=[]),
        ]
        result = ExtractionResult(
            file_name="test.pdf",
            strategy_used=ExtractionStrategy.PDFPLUMBER,
            pages=pages,
        )
        assert "Page one text" in result.full_text
        assert "Page two text" in result.full_text
        assert result.page_count == 2

    def test_truncated_text_respects_limit(self):
        long_text = "A" * 20000
        pages = [PageData(page_number=1, raw_text=long_text, tables=[])]
        result = ExtractionResult(
            file_name="test.pdf",
            strategy_used=ExtractionStrategy.PDFPLUMBER,
            pages=pages,
        )
        assert len(result.get_truncated_text(max_chars=5000)) <= 5000

    def test_char_count_is_summed(self):
        pages = [
            PageData(page_number=1, raw_text="Hello", tables=[]),
            PageData(page_number=2, raw_text="World", tables=[]),
        ]
        result = ExtractionResult(
            file_name="test.pdf",
            strategy_used=ExtractionStrategy.PDFPLUMBER,
            pages=pages,
        )
        assert result.total_chars == 10


# ---------------------------------------------------------------------------
# schemas tests
# ---------------------------------------------------------------------------

class TestAppraisalFields:
    def test_flat_dict_has_all_fields(self):
        fields = AppraisalFields()
        flat = fields.to_flat_dict()
        assert len(flat) == len(AppraisalFields.model_fields)

    def test_high_confidence_fields_filters_correctly(self):
        fields = AppraisalFields()
        fields.appraised_value = ExtractedField(
            value="$300,000", confidence=FieldConfidence.HIGH, reasoning="test"
        )
        fields.city = ExtractedField(
            value=None, confidence=FieldConfidence.MISSING, reasoning=""
        )
        high = fields.high_confidence_fields()
        assert "appraised_value" in high
        assert "city" not in high


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestNumericValidation:
    def test_valid_appraised_value_keeps_confidence(self):
        field = ExtractedField(value="$350,000", confidence=FieldConfidence.HIGH, reasoning="")
        result = _validate_numeric_field(field, "appraised_value")
        assert result.confidence == FieldConfidence.HIGH

    def test_zero_appraised_value_downgrades_confidence(self):
        field = ExtractedField(value="$0", confidence=FieldConfidence.HIGH, reasoning="")
        result = _validate_numeric_field(field, "appraised_value")
        assert result.confidence == FieldConfidence.MEDIUM
        assert "VALIDATION WARNING" in result.reasoning

    def test_future_year_built_downgrades_confidence(self):
        field = ExtractedField(value="2090", confidence=FieldConfidence.HIGH, reasoning="")
        result = _validate_numeric_field(field, "year_built")
        assert result.confidence == FieldConfidence.MEDIUM

    def test_valid_year_built_keeps_confidence(self):
        field = ExtractedField(value="1985", confidence=FieldConfidence.HIGH, reasoning="")
        result = _validate_numeric_field(field, "year_built")
        assert result.confidence == FieldConfidence.HIGH

    def test_tiny_gla_downgrades_confidence(self):
        field = ExtractedField(value="5", confidence=FieldConfidence.HIGH, reasoning="")
        result = _validate_numeric_field(field, "gross_living_area_sqft")
        assert result.confidence == FieldConfidence.MEDIUM

    def test_missing_field_is_unchanged(self):
        field = ExtractedField(value=None, confidence=FieldConfidence.MISSING, reasoning="")
        result = _validate_numeric_field(field, "appraised_value")
        assert result.confidence == FieldConfidence.MISSING


# ---------------------------------------------------------------------------
# LLMExtractor tests (mocked)
# ---------------------------------------------------------------------------

class TestLLMExtractor:
    def _make_extractor(self) -> LLMExtractor:
        return LLMExtractor(api_key="fake-key-for-tests", model="gpt-4o")

    def _mock_response(self, content_dict: dict) -> MagicMock:
        mock_msg = MagicMock()
        mock_msg.content = json.dumps(content_dict)
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    def test_extract_fields_returns_extraction_response(self):
        extractor = self._make_extractor()
        pdf_result = make_extraction_result("Property Address: 123 Main St. Value: $425,000")
        mock_response = self._mock_response(make_mock_fields())

        with patch.object(extractor.client.chat.completions, "create", return_value=mock_response):
            response = extractor.extract_fields(pdf_result)

        assert response.file_name == "test.pdf"
        assert response.fields.appraised_value.value == "$425,000"
        assert response.fields.appraised_value.confidence == FieldConfidence.HIGH
        assert isinstance(response.processing_time_seconds, float)

    def test_confidence_summary_counts_correctly(self):
        extractor = self._make_extractor()
        pdf_result = make_extraction_result("test")
        mock_response = self._mock_response(make_mock_fields())

        with patch.object(extractor.client.chat.completions, "create", return_value=mock_response):
            response = extractor.extract_fields(pdf_result)

        summary = response.confidence_summary
        assert summary["high"] >= 2   # appraised_value + address + year_built
        assert "missing" in summary

    def test_compare_documents_produces_comparison_response(self):
        extractor = self._make_extractor()
        pdf_result = make_extraction_result("test")
        mock_response = self._mock_response(make_mock_fields())

        with patch.object(extractor.client.chat.completions, "create", return_value=mock_response):
            r1 = extractor.extract_fields(pdf_result)
            r2 = extractor.extract_fields(pdf_result)
            r1.file_name = "doc1.pdf"
            r2.file_name = "doc2.pdf"

        mock_summary_response = MagicMock()
        mock_summary_response.choices[0].message.content = "Both documents are identical."

        with patch.object(extractor.client.chat.completions, "create", return_value=mock_summary_response):
            comparison = extractor.compare_documents(r1, r2)

        assert comparison.doc1_name == "doc1.pdf"
        assert comparison.doc2_name == "doc2.pdf"
        assert 0.0 <= comparison.agreement_rate <= 1.0
        assert comparison.total_fields_compared == len(AppraisalFields.model_fields)
