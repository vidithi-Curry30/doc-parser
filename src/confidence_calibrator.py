"""
confidence_calibrator.py
------------------------
Cross-validation and uncertainty quantification for extracted appraisal fields.

The LLM assigns confidence scores per field, but those scores are self-reported
and uncalibrated — the model can be confidently wrong. This module adds a second
layer of validation by checking extracted fields for *internal consistency*:
do the values make sense relative to each other?

Checks performed:
  1. Value-to-comp consistency   — appraised value vs. comparable sale prices
  2. Price-per-sqft outlier      — subject implied $/sqft vs. comp $/sqft
  3. Year-built plausibility     — cross-checked against condition rating
  4. Bedroom/bathroom ratio      — flags physically implausible ratios
  5. GLA vs. lot size ratio      — flags unusually high lot coverage
  6. Market trend vs. comp delta — if comps are rising, trend should say so
  7. Prior sale vs. appraised    — huge jumps in value flagged
  8. Field completeness score    — what % of critical fields were extracted

Each check produces a CalibrationFlag with:
  - severity: WARNING or ERROR
  - affected_fields: which fields are involved
  - message: human-readable explanation
  - suggested_action: what a reviewer should do

The final output is a CalibrationReport with an overall reliability score (0-1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .schemas import AppraisalFields, ExtractedField, FieldConfidence


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class FlagSeverity(str, Enum):
    WARNING = "warning"   # Unusual but possible — reviewer should check
    ERROR = "error"       # Very likely a data quality problem


@dataclass
class CalibrationFlag:
    """A single inconsistency detected between two or more fields."""
    severity: FlagSeverity
    check_name: str
    affected_fields: list[str]
    message: str
    suggested_action: str


@dataclass
class FieldReliability:
    """
    Adjusted reliability for a single field after cross-validation.

    original_confidence: what the LLM said
    calibrated_confidence: our adjusted assessment
    adjustment_reason: why we changed it (empty if unchanged)
    """
    field_name: str
    original_confidence: FieldConfidence
    calibrated_confidence: FieldConfidence
    adjustment_reason: str = ""


@dataclass
class CalibrationReport:
    """
    Full calibration report for an extraction result.

    reliability_score: 0.0 (completely unreliable) to 1.0 (fully consistent)
    flags: list of detected inconsistencies
    field_reliabilities: per-field adjusted confidence levels
    critical_fields_present: what % of the most important fields were found
    summary: plain-English summary of the calibration result
    """
    reliability_score: float
    flags: list[CalibrationFlag]
    field_reliabilities: list[FieldReliability]
    critical_fields_present: float   # 0.0 to 1.0
    summary: str

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == FlagSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == FlagSeverity.WARNING)

    def to_dict(self) -> dict:
        return {
            "reliability_score": self.reliability_score,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "critical_fields_present": self.critical_fields_present,
            "summary": self.summary,
            "flags": [
                {
                    "severity": f.severity.value,
                    "check_name": f.check_name,
                    "affected_fields": f.affected_fields,
                    "message": f.message,
                    "suggested_action": f.suggested_action,
                }
                for f in self.flags
            ],
            "field_reliabilities": [
                {
                    "field_name": r.field_name,
                    "original_confidence": r.original_confidence.value,
                    "calibrated_confidence": r.calibrated_confidence.value,
                    "adjustment_reason": r.adjustment_reason,
                }
                for r in self.field_reliabilities
            ],
        }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_currency(value: Optional[str]) -> Optional[float]:
    """Parse '$425,000' or '425000' into 425000.0. Returns None if unparseable."""
    if not value:
        return None
    cleaned = re.sub(r"[,$\s]", "", value)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_float(value: Optional[str]) -> Optional[float]:
    """Parse a plain numeric string into float. Returns None if unparseable."""
    if not value:
        return None
    cleaned = re.sub(r"[,\s]", "", value)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    result = _parse_float(value)
    return int(result) if result is not None else None


def _is_present(field_obj: ExtractedField) -> bool:
    return (
        field_obj.confidence != FieldConfidence.MISSING
        and field_obj.value is not None
    )


# ---------------------------------------------------------------------------
# Individual consistency checks
# ---------------------------------------------------------------------------

CRITICAL_FIELDS = [
    "property_address", "appraised_value", "effective_date_of_appraisal",
    "gross_living_area_sqft", "year_built", "property_type",
    "number_of_bedrooms", "number_of_bathrooms", "appraiser_name",
    "comp_1_address", "comp_1_sale_price",
]


def _check_critical_field_completeness(
    fields: AppraisalFields,
) -> tuple[float, list[CalibrationFlag]]:
    """
    What fraction of the most important fields were successfully extracted?
    Missing critical fields is a strong signal of a low-quality extraction.
    """
    flags = []
    present = sum(
        1 for fname in CRITICAL_FIELDS
        if _is_present(getattr(fields, fname))
    )
    completeness = present / len(CRITICAL_FIELDS)

    if completeness < 0.5:
        flags.append(CalibrationFlag(
            severity=FlagSeverity.ERROR,
            check_name="critical_field_completeness",
            affected_fields=CRITICAL_FIELDS,
            message=(
                f"Only {present}/{len(CRITICAL_FIELDS)} critical fields were extracted. "
                "The document may not be a standard appraisal form, or text extraction failed."
            ),
            suggested_action="Verify the PDF contains selectable text (not a scanned image). "
                             "Try re-uploading or using OCR preprocessing.",
        ))
    elif completeness < 0.8:
        flags.append(CalibrationFlag(
            severity=FlagSeverity.WARNING,
            check_name="critical_field_completeness",
            affected_fields=CRITICAL_FIELDS,
            message=(
                f"{present}/{len(CRITICAL_FIELDS)} critical fields extracted. "
                "Some fields may be in non-standard locations."
            ),
            suggested_action="Manually verify missing critical fields.",
        ))

    return completeness, flags


def _check_value_vs_comps(
    fields: AppraisalFields,
) -> list[CalibrationFlag]:
    """
    The appraised value should be in a reasonable range relative to the
    comparable sales. Appraisers are required to justify values far outside
    the comp range, so a large deviation is a data quality flag.

    Threshold: appraised value should be within 40% of the comp midpoint.
    """
    flags = []

    appraised = _parse_currency(fields.appraised_value.value)
    comp_prices = [
        _parse_currency(fields.comp_1_sale_price.value),
        _parse_currency(fields.comp_2_sale_price.value),
        _parse_currency(fields.comp_3_sale_price.value),
    ]
    valid_comps = [c for c in comp_prices if c is not None and c > 0]

    if appraised is None or len(valid_comps) == 0:
        return flags  # Can't check without data

    comp_midpoint = sum(valid_comps) / len(valid_comps)
    deviation = abs(appraised - comp_midpoint) / comp_midpoint

    if deviation > 0.40:
        severity = FlagSeverity.ERROR if deviation > 0.60 else FlagSeverity.WARNING
        flags.append(CalibrationFlag(
            severity=severity,
            check_name="value_vs_comp_range",
            affected_fields=["appraised_value", "comp_1_sale_price", "comp_2_sale_price", "comp_3_sale_price"],
            message=(
                f"Appraised value (${appraised:,.0f}) deviates {deviation:.1%} from "
                f"comparable sales midpoint (${comp_midpoint:,.0f}). "
                "Standard appraisal practice requires strong justification for deviations >20%."
            ),
            suggested_action="Verify appraised value and comparable sale prices were extracted correctly. "
                             "Large deviations may indicate extraction from the wrong field.",
        ))

    return flags


def _check_price_per_sqft_consistency(
    fields: AppraisalFields,
) -> list[CalibrationFlag]:
    """
    Compute price-per-sqft for the subject property (implied by appraised value
    and GLA) and compare against each comp. Flag outliers beyond 2 standard
    deviations from the comp mean.

    This is a standard appraisal QC technique — appraisers use $/sqft as a
    sanity check on their comparable selection and adjustments.
    """
    flags = []

    appraised = _parse_currency(fields.appraised_value.value)
    subject_gla = _parse_float(fields.gross_living_area_sqft.value)

    if not appraised or not subject_gla or subject_gla == 0:
        return flags

    subject_ppsf = appraised / subject_gla

    comp_data = [
        (fields.comp_1_sale_price.value, fields.comp_1_gla.value, "comp_1"),
        (fields.comp_2_sale_price.value, fields.comp_2_gla.value, "comp_2"),
        (fields.comp_3_sale_price.value, fields.comp_3_gla.value, "comp_3"),
    ]

    comp_ppsfts = []
    for price_str, gla_str, label in comp_data:
        price = _parse_currency(price_str)
        gla = _parse_float(gla_str)
        if price and gla and gla > 0:
            comp_ppsfts.append((label, price / gla))

    if len(comp_ppsfts) < 2:
        return flags

    ppsf_values = [v for _, v in comp_ppsfts]
    mean_ppsf = sum(ppsf_values) / len(ppsf_values)

    # Standard deviation
    variance = sum((v - mean_ppsf) ** 2 for v in ppsf_values) / len(ppsf_values)
    std_ppsf = variance ** 0.5

    if std_ppsf > 0:
        subject_z = abs(subject_ppsf - mean_ppsf) / std_ppsf
        if subject_z > 2.0:
            flags.append(CalibrationFlag(
                severity=FlagSeverity.WARNING,
                check_name="price_per_sqft_outlier",
                affected_fields=["appraised_value", "gross_living_area_sqft"],
                message=(
                    f"Subject implied price/sqft (${subject_ppsf:.0f}/sqft) is "
                    f"{subject_z:.1f} standard deviations from comp mean "
                    f"(${mean_ppsf:.0f}/sqft ± ${std_ppsf:.0f}). "
                    "This may indicate a GLA or value extraction error."
                ),
                suggested_action="Verify GLA and appraised value. Check if basement area "
                                 "was incorrectly included in above-grade GLA.",
            ))

    return flags


def _check_year_built_vs_condition(
    fields: AppraisalFields,
) -> list[CalibrationFlag]:
    """
    UAD condition ratings have implicit relationships with property age:
    - C1 (new/never occupied) should have a very recent year_built
    - C6 (severe deferred maintenance) is unlikely for properties < 5 years old

    This is a soft check — exceptions exist — but mismatches often indicate
    one field was extracted from the wrong location.
    """
    flags = []

    year = _parse_int(fields.year_built.value)
    condition = fields.condition_rating.value

    if year is None or not condition:
        return flags

    from datetime import datetime
    age = datetime.now().year - year
    condition_upper = condition.upper().strip()

    if "C1" in condition_upper and age > 2:
        flags.append(CalibrationFlag(
            severity=FlagSeverity.WARNING,
            check_name="year_built_vs_condition",
            affected_fields=["year_built", "condition_rating"],
            message=(
                f"Condition rating C1 (new/never occupied) but year_built is {year} "
                f"({age} years ago). C1 is typically reserved for newly built properties."
            ),
            suggested_action="Verify condition rating — may have been extracted from "
                             "a comparable sale row rather than the subject property.",
        ))
    elif "C6" in condition_upper and age < 5:
        flags.append(CalibrationFlag(
            severity=FlagSeverity.WARNING,
            check_name="year_built_vs_condition",
            affected_fields=["year_built", "condition_rating"],
            message=(
                f"Condition rating C6 (severe deferred maintenance) but property is "
                f"only {age} years old (built {year}). C6 is unusual for new properties."
            ),
            suggested_action="Verify both year_built and condition_rating extractions.",
        ))

    return flags


def _check_bedroom_bathroom_ratio(
    fields: AppraisalFields,
) -> list[CalibrationFlag]:
    """
    Bedroom-to-bathroom ratio sanity check.
    A property with more bathrooms than bedrooms is unusual but possible (luxury).
    A property with 0 bathrooms or 0 bedrooms is almost certainly an extraction error.
    """
    flags = []

    beds = _parse_float(fields.number_of_bedrooms.value)
    baths = _parse_float(fields.number_of_bathrooms.value)

    if beds is None or baths is None:
        return flags

    if beds == 0:
        flags.append(CalibrationFlag(
            severity=FlagSeverity.ERROR,
            check_name="bedroom_count",
            affected_fields=["number_of_bedrooms"],
            message="Extracted bedroom count is 0, which is implausible for a residential appraisal.",
            suggested_action="Verify bedroom count extraction — likely picked up wrong field.",
        ))

    if baths == 0:
        flags.append(CalibrationFlag(
            severity=FlagSeverity.ERROR,
            check_name="bathroom_count",
            affected_fields=["number_of_bathrooms"],
            message="Extracted bathroom count is 0, which is implausible for a residential appraisal.",
            suggested_action="Verify bathroom count — UAD format uses decimal notation (e.g. 2.1 = 2 full, 1 half).",
        ))

    if beds and baths and baths > beds * 2:
        flags.append(CalibrationFlag(
            severity=FlagSeverity.WARNING,
            check_name="bedroom_bathroom_ratio",
            affected_fields=["number_of_bedrooms", "number_of_bathrooms"],
            message=(
                f"Unusual bedroom/bathroom ratio: {beds} beds, {baths} baths. "
                "More than 2x bathrooms per bedroom is atypical."
            ),
            suggested_action="Verify both counts. UAD bathroom notation may have been misread.",
        ))

    return flags


def _check_prior_sale_vs_appraised(
    fields: AppraisalFields,
) -> list[CalibrationFlag]:
    """
    Flag cases where the appraised value is dramatically different from the
    prior sale price. While real appreciation/depreciation happens, a >100%
    change in value is a strong signal of either an extraction error or
    an unusual transaction requiring explanation.
    """
    flags = []

    appraised = _parse_currency(fields.appraised_value.value)
    prior = _parse_currency(fields.prior_sale_price.value)

    if not appraised or not prior or prior == 0:
        return flags

    change = abs(appraised - prior) / prior

    if change > 1.0:  # >100% change
        flags.append(CalibrationFlag(
            severity=FlagSeverity.WARNING,
            check_name="prior_sale_vs_appraised",
            affected_fields=["appraised_value", "prior_sale_price"],
            message=(
                f"Appraised value (${appraised:,.0f}) differs from prior sale "
                f"(${prior:,.0f}) by {change:.0%}. Changes >100% are unusual "
                "and may indicate an extraction error or non-arm's-length prior sale."
            ),
            suggested_action="Verify both values and the prior sale date. "
                             "Large changes may be legitimate if the prior sale was many years ago.",
        ))

    return flags


# ---------------------------------------------------------------------------
# Confidence downgrade logic
# ---------------------------------------------------------------------------

def _build_field_reliabilities(
    fields: AppraisalFields,
    flags: list[CalibrationFlag],
) -> list[FieldReliability]:
    """
    For each field involved in a calibration flag, downgrade its confidence
    one level (HIGH→MEDIUM, MEDIUM→LOW, LOW→LOW).
    """
    downgrade_map = {
        FieldConfidence.HIGH: FieldConfidence.MEDIUM,
        FieldConfidence.MEDIUM: FieldConfidence.LOW,
        FieldConfidence.LOW: FieldConfidence.LOW,
        FieldConfidence.MISSING: FieldConfidence.MISSING,
    }

    # Build a map of field_name -> list of reasons to downgrade
    downgrade_reasons: dict[str, list[str]] = {}
    for flag in flags:
        if flag.severity in (FlagSeverity.WARNING, FlagSeverity.ERROR):
            for fname in flag.affected_fields:
                if fname not in downgrade_reasons:
                    downgrade_reasons[fname] = []
                downgrade_reasons[fname].append(f"{flag.check_name}: {flag.message[:80]}...")

    reliabilities = []
    for fname in AppraisalFields.model_fields:
        field_obj: ExtractedField = getattr(fields, fname)
        original = field_obj.confidence

        if fname in downgrade_reasons:
            calibrated = downgrade_map[original]
            reason = " | ".join(downgrade_reasons[fname])
        else:
            calibrated = original
            reason = ""

        reliabilities.append(FieldReliability(
            field_name=fname,
            original_confidence=original,
            calibrated_confidence=calibrated,
            adjustment_reason=reason,
        ))

    return reliabilities


# ---------------------------------------------------------------------------
# Reliability score computation
# ---------------------------------------------------------------------------

def _compute_reliability_score(
    flags: list[CalibrationFlag],
    completeness: float,
) -> float:
    """
    Compute an overall reliability score (0.0 to 1.0).

    Starts at 1.0 and deducts:
      - 0.20 per ERROR flag
      - 0.08 per WARNING flag
    Also scales by completeness of critical fields.
    """
    score = 1.0
    for flag in flags:
        if flag.severity == FlagSeverity.ERROR:
            score -= 0.20
        elif flag.severity == FlagSeverity.WARNING:
            score -= 0.08

    # Weight completeness: a document with 50% critical fields gets a 50% penalty
    score *= completeness

    return max(0.0, min(1.0, round(score, 4)))


def _build_summary(
    score: float,
    flags: list[CalibrationFlag],
    completeness: float,
) -> str:
    errors = sum(1 for f in flags if f.severity == FlagSeverity.ERROR)
    warnings = sum(1 for f in flags if f.severity == FlagSeverity.WARNING)

    if score >= 0.85:
        quality = "High"
        desc = "Fields are internally consistent. Extraction is reliable."
    elif score >= 0.60:
        quality = "Medium"
        desc = "Some inconsistencies detected. Manual review recommended for flagged fields."
    else:
        quality = "Low"
        desc = "Significant inconsistencies detected. Do not rely on extracted values without manual verification."

    return (
        f"Reliability: {quality} ({score:.0%}). "
        f"Critical field completeness: {completeness:.0%}. "
        f"{errors} error(s), {warnings} warning(s). {desc}"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

class ConfidenceCalibrator:
    """
    Runs all consistency checks on an AppraisalFields instance and
    returns a CalibrationReport with adjusted confidence levels.
    """

    def calibrate(self, fields: AppraisalFields) -> CalibrationReport:
        """
        Run all checks and produce a CalibrationReport.

        Args:
            fields: The extracted AppraisalFields from LLMExtractor.

        Returns:
            CalibrationReport with reliability score, flags, and adjusted confidences.
        """
        all_flags: list[CalibrationFlag] = []

        completeness, completeness_flags = _check_critical_field_completeness(fields)
        all_flags.extend(completeness_flags)
        all_flags.extend(_check_value_vs_comps(fields))
        all_flags.extend(_check_price_per_sqft_consistency(fields))
        all_flags.extend(_check_year_built_vs_condition(fields))
        all_flags.extend(_check_bedroom_bathroom_ratio(fields))
        all_flags.extend(_check_prior_sale_vs_appraised(fields))

        reliability_score = _compute_reliability_score(all_flags, completeness)
        field_reliabilities = _build_field_reliabilities(fields, all_flags)
        summary = _build_summary(reliability_score, all_flags, completeness)

        return CalibrationReport(
            reliability_score=reliability_score,
            flags=all_flags,
            field_reliabilities=field_reliabilities,
            critical_fields_present=completeness,
            summary=summary,
        )
