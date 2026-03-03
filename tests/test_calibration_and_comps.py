"""
tests/test_calibration_and_comps.py
------------------------------------
Unit tests for the confidence calibrator and comparable sales adjustment engine.
No API calls — all tests use synthetic AppraisalFields instances.
"""

from __future__ import annotations

import pytest

from src.schemas import AppraisalFields, ExtractedField, FieldConfidence
from src.confidence_calibrator import (
    ConfidenceCalibrator,
    FlagSeverity,
    _check_value_vs_comps,
    _check_price_per_sqft_consistency,
    _check_year_built_vs_condition,
    _check_bedroom_bathroom_ratio,
    _check_prior_sale_vs_appraised,
    _check_critical_field_completeness,
)
from src.comp_engine import (
    CompAdjustmentEngine,
    ComparableSale,
    _parse_currency,
    _parse_float,
    _ppsf_analysis,
    _estimate_gla_adjustment_rate,
    _check_bracketing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def high(value: str) -> ExtractedField:
    return ExtractedField(value=value, confidence=FieldConfidence.HIGH, reasoning="test")

def missing() -> ExtractedField:
    return ExtractedField(value=None, confidence=FieldConfidence.MISSING, reasoning="")

def make_full_fields(
    appraised_value="$400,000",
    gla="2000",
    year_built="1990",
    condition="C3",
    bedrooms="3",
    bathrooms="2",
    prior_sale="$200,000",
    comp1_price="$390,000", comp1_gla="1950",
    comp2_price="$410,000", comp2_gla="2050",
    comp3_price="$405,000", comp3_gla="2000",
) -> AppraisalFields:
    """Build a realistic AppraisalFields with sensible defaults."""
    f = AppraisalFields()
    f.appraised_value = high(appraised_value)
    f.gross_living_area_sqft = high(gla)
    f.year_built = high(year_built)
    f.condition_rating = high(condition)
    f.number_of_bedrooms = high(bedrooms)
    f.number_of_bathrooms = high(bathrooms)
    f.prior_sale_price = high(prior_sale)
    f.property_address = high("123 Main St")
    f.effective_date_of_appraisal = high("2024-01-15")
    f.property_type = high("Single Family")
    f.appraiser_name = high("Jane Smith")
    f.comp_1_address = high("100 Oak Ave")
    f.comp_1_sale_price = high(comp1_price)
    f.comp_1_gla = high(comp1_gla)
    f.comp_2_address = high("200 Elm St")
    f.comp_2_sale_price = high(comp2_price)
    f.comp_2_gla = high(comp2_gla)
    f.comp_3_address = high("300 Pine Rd")
    f.comp_3_sale_price = high(comp3_price)
    f.comp_3_gla = high(comp3_gla)
    return f


# ---------------------------------------------------------------------------
# Parsing helper tests
# ---------------------------------------------------------------------------

class TestParsers:
    def test_parse_currency_with_dollar_sign(self):
        assert _parse_currency("$425,000") == 425000.0

    def test_parse_currency_plain(self):
        assert _parse_currency("425000") == 425000.0

    def test_parse_currency_none(self):
        assert _parse_currency(None) is None

    def test_parse_currency_invalid(self):
        assert _parse_currency("N/A") is None

    def test_parse_float_with_comma(self):
        assert _parse_float("2,150") == 2150.0

    def test_parse_float_plain(self):
        assert _parse_float("1800") == 1800.0


# ---------------------------------------------------------------------------
# Confidence calibrator tests
# ---------------------------------------------------------------------------

class TestValueVsComps:
    def test_no_flag_when_value_within_range(self):
        fields = make_full_fields(
            appraised_value="$400,000",
            comp1_price="$390,000", comp2_price="$410,000", comp3_price="$405,000"
        )
        flags = _check_value_vs_comps(fields)
        assert len(flags) == 0

    def test_warning_when_value_40pct_above_comps(self):
        fields = make_full_fields(
            appraised_value="$600,000",
            comp1_price="$390,000", comp2_price="$410,000", comp3_price="$405,000"
        )
        flags = _check_value_vs_comps(fields)
        assert len(flags) == 1
        assert flags[0].severity in (FlagSeverity.WARNING, FlagSeverity.ERROR)

    def test_no_flag_when_comps_missing(self):
        fields = AppraisalFields()
        fields.appraised_value = high("$400,000")
        flags = _check_value_vs_comps(fields)
        assert len(flags) == 0


class TestYearBuiltVsCondition:
    def test_c1_new_property_no_flag(self):
        from datetime import datetime
        current_year = datetime.now().year
        fields = make_full_fields(year_built=str(current_year), condition="C1")
        flags = _check_year_built_vs_condition(fields)
        assert len(flags) == 0

    def test_c1_old_property_raises_warning(self):
        fields = make_full_fields(year_built="1985", condition="C1")
        flags = _check_year_built_vs_condition(fields)
        assert len(flags) == 1
        assert flags[0].severity == FlagSeverity.WARNING

    def test_c6_new_property_raises_warning(self):
        from datetime import datetime
        current_year = datetime.now().year
        fields = make_full_fields(year_built=str(current_year), condition="C6")
        flags = _check_year_built_vs_condition(fields)
        assert len(flags) == 1

    def test_normal_condition_no_flag(self):
        fields = make_full_fields(year_built="1990", condition="C3")
        flags = _check_year_built_vs_condition(fields)
        assert len(flags) == 0


class TestBedroomBathroomRatio:
    def test_normal_ratio_no_flag(self):
        fields = make_full_fields(bedrooms="3", bathrooms="2")
        flags = _check_bedroom_bathroom_ratio(fields)
        assert len(flags) == 0

    def test_zero_bedrooms_raises_error(self):
        fields = make_full_fields(bedrooms="0", bathrooms="2")
        flags = _check_bedroom_bathroom_ratio(fields)
        assert any(f.severity == FlagSeverity.ERROR for f in flags)

    def test_zero_bathrooms_raises_error(self):
        fields = make_full_fields(bedrooms="3", bathrooms="0")
        flags = _check_bedroom_bathroom_ratio(fields)
        assert any(f.severity == FlagSeverity.ERROR for f in flags)

    def test_too_many_bathrooms_raises_warning(self):
        fields = make_full_fields(bedrooms="2", bathrooms="10")
        flags = _check_bedroom_bathroom_ratio(fields)
        assert any(f.severity == FlagSeverity.WARNING for f in flags)


class TestPriorSaleVsAppraised:
    def test_reasonable_appreciation_no_flag(self):
        fields = make_full_fields(appraised_value="$400,000", prior_sale="$300,000")
        flags = _check_prior_sale_vs_appraised(fields)
        assert len(flags) == 0

    def test_massive_increase_raises_warning(self):
        fields = make_full_fields(appraised_value="$900,000", prior_sale="$100,000")
        flags = _check_prior_sale_vs_appraised(fields)
        assert len(flags) == 1
        assert flags[0].severity == FlagSeverity.WARNING

    def test_missing_prior_sale_no_flag(self):
        fields = make_full_fields(appraised_value="$400,000")
        fields.prior_sale_price = missing()
        flags = _check_prior_sale_vs_appraised(fields)
        assert len(flags) == 0


class TestCriticalFieldCompleteness:
    def test_full_fields_high_completeness(self):
        fields = make_full_fields()
        completeness, flags = _check_critical_field_completeness(fields)
        assert completeness >= 0.9
        assert len(flags) == 0

    def test_empty_fields_low_completeness(self):
        fields = AppraisalFields()
        completeness, flags = _check_critical_field_completeness(fields)
        assert completeness == 0.0
        assert len(flags) == 1
        assert flags[0].severity == FlagSeverity.ERROR


class TestCalibrationReport:
    def test_clean_document_high_reliability(self):
        calibrator = ConfidenceCalibrator()
        fields = make_full_fields()
        report = calibrator.calibrate(fields)
        assert report.reliability_score >= 0.7
        assert report.error_count == 0

    def test_bad_document_low_reliability(self):
        calibrator = ConfidenceCalibrator()
        fields = make_full_fields(
            appraised_value="$1,000,000",   # Way above comps
            comp1_price="$200,000",
            comp2_price="$210,000",
            comp3_price="$205,000",
            bedrooms="0",                   # Invalid
            year_built="1850",
        )
        report = calibrator.calibrate(fields)
        assert report.reliability_score < 0.8
        assert len(report.flags) > 0

    def test_report_has_all_fields_in_reliabilities(self):
        calibrator = ConfidenceCalibrator()
        fields = make_full_fields()
        report = calibrator.calibrate(fields)
        reliability_names = {r.field_name for r in report.field_reliabilities}
        assert set(AppraisalFields.model_fields.keys()) == reliability_names

    def test_to_dict_is_serializable(self):
        calibrator = ConfidenceCalibrator()
        fields = make_full_fields()
        report = calibrator.calibrate(fields)
        d = report.to_dict()
        assert "reliability_score" in d
        assert "flags" in d
        assert isinstance(d["flags"], list)


# ---------------------------------------------------------------------------
# Comp engine tests
# ---------------------------------------------------------------------------

class TestBracketingCheck:
    def test_properly_bracketed(self):
        comps = [
            ComparableSale("comp_1", "addr1", 380000, 1900),
            ComparableSale("comp_2", "addr2", 420000, 2100),
            ComparableSale("comp_3", "addr3", 400000, 2000),
        ]
        result = _check_bracketing(400000, comps)
        assert result.is_bracketed is True

    def test_all_comps_above_not_bracketed(self):
        comps = [
            ComparableSale("comp_1", "addr1", 450000, 2000),
            ComparableSale("comp_2", "addr2", 460000, 2100),
            ComparableSale("comp_3", "addr3", 470000, 2200),
        ]
        result = _check_bracketing(400000, comps)
        assert result.is_bracketed is False
        assert result.has_comp_above is True
        assert result.has_comp_below is False

    def test_all_comps_below_not_bracketed(self):
        comps = [
            ComparableSale("comp_1", "addr1", 350000, 1900),
            ComparableSale("comp_2", "addr2", 360000, 1950),
            ComparableSale("comp_3", "addr3", 370000, 2000),
        ]
        result = _check_bracketing(400000, comps)
        assert result.is_bracketed is False
        assert result.has_comp_below is True
        assert result.has_comp_above is False


class TestGLARate:
    def test_rate_computed_from_two_comps(self):
        comps = [
            ComparableSale("comp_1", "addr", 400000, 2000),
            ComparableSale("comp_2", "addr", 500000, 2500),
        ]
        rate = _estimate_gla_adjustment_rate(comps)
        assert rate is not None
        assert rate > 0

    def test_returns_none_with_one_comp(self):
        comps = [ComparableSale("comp_1", "addr", 400000, 2000)]
        rate = _estimate_gla_adjustment_rate(comps)
        assert rate is None


class TestCompAdjustmentEngine:
    def test_full_analysis_produces_report(self):
        engine = CompAdjustmentEngine()
        fields = make_full_fields()
        report = engine.analyze(fields)
        assert report.valid_comp_count == 3
        assert report.comp_ppsf_mean is not None
        assert report.comp_quality_score >= 0.0

    def test_properly_bracketed_document(self):
        engine = CompAdjustmentEngine()
        fields = make_full_fields(
            appraised_value="$400,000",
            comp1_price="$390,000",
            comp2_price="$410,000",
            comp3_price="$405,000",
        )
        report = engine.analyze(fields)
        assert report.bracketing is not None
        assert report.bracketing.is_bracketed is True

    def test_unbr_acketed_document_flagged(self):
        engine = CompAdjustmentEngine()
        fields = make_full_fields(
            appraised_value="$400,000",
            comp1_price="$450,000",
            comp2_price="$460,000",
            comp3_price="$470,000",
        )
        report = engine.analyze(fields)
        assert report.bracketing.is_bracketed is False
        assert len(report.warnings) > 0

    def test_no_comps_graceful(self):
        engine = CompAdjustmentEngine()
        fields = AppraisalFields()
        fields.appraised_value = high("$400,000")
        fields.gross_living_area_sqft = high("2000")
        report = engine.analyze(fields)
        assert report.valid_comp_count == 0
        assert len(report.warnings) > 0

    def test_to_dict_is_serializable(self):
        engine = CompAdjustmentEngine()
        fields = make_full_fields()
        report = engine.analyze(fields)
        d = report.to_dict()
        assert "comp_quality_score" in d
        assert "gla_adjustment_analysis" in d
        assert "bracketing" in d
        assert isinstance(d["comparables"], list)
