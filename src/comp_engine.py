"""
comp_engine.py
--------------
Comparable Sales Adjustment Engine for residential appraisals.

In real appraisal practice, appraisers make dollar adjustments to comparable
sales to account for differences between the comp and the subject property
(e.g. if comp has 3 beds and subject has 4, a positive adjustment is made).

This module automates the quantitative analysis that appraisers do manually:

  1. Price-per-sqft analysis
     - Compute $/sqft for subject (implied) and each comp
     - Identify which comps are most similar in size to the subject

  2. GLA adjustment estimation
     - Estimate a $/sqft adjustment rate from the comp data
     - Apply it to GLA differences between subject and each comp

  3. Value bracketing check
     - UAD guidelines require that comps "bracket" the subject value
       (at least one comp above and one below the appraised value)
     - Flag if bracketing is absent

  4. Comp selection quality score
     - Score how well the selected comps support the appraised value
     - Based on: price proximity, GLA similarity, number of valid comps

  5. Adjusted value range
     - After GLA adjustments, compute the implied value range supported
       by the comps and flag if appraised value falls outside it

Output: CompAdjustmentReport — a structured report with all analysis results,
designed to be embedded in the main ExtractionResponse.

Note: This is a *quantitative analysis tool*, not a replacement for a
licensed appraiser. It uses only the data extracted from the document.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Optional

from .schemas import AppraisalFields


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ComparableSale:
    """Parsed data for a single comparable sale."""
    label: str              # "comp_1", "comp_2", "comp_3"
    address: Optional[str]
    sale_price: Optional[float]
    gla_sqft: Optional[float]
    price_per_sqft: Optional[float] = field(init=False)

    def __post_init__(self):
        if self.sale_price and self.gla_sqft and self.gla_sqft > 0:
            self.price_per_sqft = round(self.sale_price / self.gla_sqft, 2)
        else:
            self.price_per_sqft = None


@dataclass
class GLAAdjustment:
    """
    Estimated GLA adjustment for one comparable sale.

    The adjustment answers: "How much should we add/subtract from this
    comp's sale price to account for its GLA difference from the subject?"

    Formula: adjustment = gla_difference_sqft × estimated_rate_per_sqft
    """
    comp_label: str
    comp_sale_price: float
    comp_gla: float
    subject_gla: float
    gla_difference_sqft: float      # positive = subject is larger
    estimated_rate_per_sqft: float  # $/sqft adjustment rate
    raw_adjustment: float           # = gla_difference × rate
    adjusted_comp_value: float      # comp price + adjustment


@dataclass
class BracketingCheck:
    """
    UAD bracketing requirement check.
    At least one comp should be priced above and one below the subject value.
    """
    appraised_value: float
    comp_prices: list[float]
    has_comp_above: bool
    has_comp_below: bool
    is_bracketed: bool
    message: str


@dataclass
class CompAdjustmentReport:
    """Full comparable sales analysis report."""

    # Subject property data
    subject_appraised_value: Optional[float]
    subject_gla_sqft: Optional[float]
    subject_implied_ppsf: Optional[float]

    # Parsed comps
    comparables: list[ComparableSale]
    valid_comp_count: int

    # Price-per-sqft analysis
    comp_ppsf_mean: Optional[float]
    comp_ppsf_std: Optional[float]
    comp_ppsf_min: Optional[float]
    comp_ppsf_max: Optional[float]
    subject_ppsf_vs_comp_mean_pct: Optional[float]  # how far above/below mean

    # GLA adjustment analysis
    estimated_gla_rate_per_sqft: Optional[float]
    gla_adjustments: list[GLAAdjustment]
    adjusted_value_range_low: Optional[float]
    adjusted_value_range_high: Optional[float]
    appraised_value_in_range: Optional[bool]

    # Bracketing
    bracketing: Optional[BracketingCheck]

    # Overall comp selection quality
    comp_quality_score: float   # 0.0 to 1.0
    comp_quality_label: str     # "Strong", "Adequate", "Weak"

    # Summary
    summary: str
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "subject_appraised_value": self.subject_appraised_value,
            "subject_gla_sqft": self.subject_gla_sqft,
            "subject_implied_ppsf": self.subject_implied_ppsf,
            "valid_comp_count": self.valid_comp_count,
            "comparables": [
                {
                    "label": c.label,
                    "address": c.address,
                    "sale_price": c.sale_price,
                    "gla_sqft": c.gla_sqft,
                    "price_per_sqft": c.price_per_sqft,
                }
                for c in self.comparables
            ],
            "price_per_sqft_analysis": {
                "comp_mean": self.comp_ppsf_mean,
                "comp_std": self.comp_ppsf_std,
                "comp_min": self.comp_ppsf_min,
                "comp_max": self.comp_ppsf_max,
                "subject_vs_mean_pct": self.subject_ppsf_vs_comp_mean_pct,
            },
            "gla_adjustment_analysis": {
                "estimated_rate_per_sqft": self.estimated_gla_rate_per_sqft,
                "adjusted_value_range": {
                    "low": self.adjusted_value_range_low,
                    "high": self.adjusted_value_range_high,
                },
                "appraised_value_in_range": self.appraised_value_in_range,
                "adjustments": [
                    {
                        "comp": a.comp_label,
                        "comp_sale_price": a.comp_sale_price,
                        "gla_difference_sqft": a.gla_difference_sqft,
                        "adjustment": a.raw_adjustment,
                        "adjusted_comp_value": a.adjusted_comp_value,
                    }
                    for a in self.gla_adjustments
                ],
            },
            "bracketing": {
                "is_bracketed": self.bracketing.is_bracketed if self.bracketing else None,
                "has_comp_above": self.bracketing.has_comp_above if self.bracketing else None,
                "has_comp_below": self.bracketing.has_comp_below if self.bracketing else None,
                "message": self.bracketing.message if self.bracketing else None,
            },
            "comp_quality_score": self.comp_quality_score,
            "comp_quality_label": self.comp_quality_label,
            "summary": self.summary,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_currency(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    cleaned = re.sub(r"[,$\s]", "", value)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_float(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    cleaned = re.sub(r"[,\s]", "", value)
    try:
        return float(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _parse_comparables(fields: AppraisalFields) -> list[ComparableSale]:
    """Extract and parse all three comparable sales from the fields."""
    raw = [
        ("comp_1", fields.comp_1_address.value, fields.comp_1_sale_price.value, fields.comp_1_gla.value),
        ("comp_2", fields.comp_2_address.value, fields.comp_2_sale_price.value, fields.comp_2_gla.value),
        ("comp_3", fields.comp_3_address.value, fields.comp_3_sale_price.value, fields.comp_3_gla.value),
    ]
    comps = []
    for label, addr, price_str, gla_str in raw:
        comps.append(ComparableSale(
            label=label,
            address=addr,
            sale_price=_parse_currency(price_str),
            gla_sqft=_parse_float(gla_str),
        ))
    return comps


def _ppsf_analysis(
    subject_value: Optional[float],
    subject_gla: Optional[float],
    comps: list[ComparableSale],
) -> dict:
    """Compute price-per-sqft statistics across comps and subject."""
    valid = [c for c in comps if c.price_per_sqft is not None]

    if not valid:
        return {
            "mean": None, "std": None, "min": None, "max": None,
            "subject_implied": None, "subject_vs_mean_pct": None,
        }

    ppsf_values = [c.price_per_sqft for c in valid]
    mean_ppsf = statistics.mean(ppsf_values)
    std_ppsf = statistics.stdev(ppsf_values) if len(ppsf_values) > 1 else 0.0

    subject_implied = None
    subject_vs_mean_pct = None
    if subject_value and subject_gla and subject_gla > 0:
        subject_implied = subject_value / subject_gla
        subject_vs_mean_pct = (subject_implied - mean_ppsf) / mean_ppsf

    return {
        "mean": round(mean_ppsf, 2),
        "std": round(std_ppsf, 2),
        "min": round(min(ppsf_values), 2),
        "max": round(max(ppsf_values), 2),
        "subject_implied": round(subject_implied, 2) if subject_implied else None,
        "subject_vs_mean_pct": round(subject_vs_mean_pct, 4) if subject_vs_mean_pct is not None else None,
    }


def _estimate_gla_adjustment_rate(comps: list[ComparableSale]) -> Optional[float]:
    """
    Estimate the market rate for GLA adjustments ($/sqft) from the comp data.

    Method: We use the relationship between $/sqft and GLA across comps.
    Larger homes tend to have lower $/sqft (diminishing returns), so we
    use a simple weighted average of the comp $/sqft values as our rate.

    In a full appraisal system, this would use paired sales analysis or
    regression. Here we use the comp $/sqft mean as a reasonable proxy.

    Returns None if fewer than 2 comps have valid data.
    """
    valid = [c for c in comps if c.price_per_sqft is not None]
    if len(valid) < 2:
        return None

    # Weight by inverse GLA so larger comps don't dominate
    weighted_sum = sum(c.price_per_sqft / c.gla_sqft for c in valid)
    weight_total = sum(1 / c.gla_sqft for c in valid)
    rate = weighted_sum / weight_total if weight_total > 0 else None

    # Apply a 25% haircut — market GLA adjustments are typically less than
    # the full $/sqft rate because location and other factors dominate
    return round(rate * 0.25, 2) if rate else None


def _compute_gla_adjustments(
    subject_gla: float,
    comps: list[ComparableSale],
    rate_per_sqft: float,
) -> list[GLAAdjustment]:
    """
    For each comp with valid data, compute the GLA adjustment and adjusted value.
    """
    adjustments = []
    for comp in comps:
        if comp.sale_price is None or comp.gla_sqft is None:
            continue
        diff = subject_gla - comp.gla_sqft   # positive = subject is bigger
        adjustment = diff * rate_per_sqft
        adjustments.append(GLAAdjustment(
            comp_label=comp.label,
            comp_sale_price=comp.sale_price,
            comp_gla=comp.gla_sqft,
            subject_gla=subject_gla,
            gla_difference_sqft=round(diff, 0),
            estimated_rate_per_sqft=rate_per_sqft,
            raw_adjustment=round(adjustment, 0),
            adjusted_comp_value=round(comp.sale_price + adjustment, 0),
        ))
    return adjustments


def _check_bracketing(
    appraised_value: float,
    comps: list[ComparableSale],
) -> BracketingCheck:
    """
    UAD requires that comp sales 'bracket' the subject value — at least one
    comp sold for more than the appraised value, and at least one for less.
    This helps demonstrate market support for the value conclusion.
    """
    valid_prices = [c.sale_price for c in comps if c.sale_price is not None]

    has_above = any(p > appraised_value for p in valid_prices)
    has_below = any(p < appraised_value for p in valid_prices)
    is_bracketed = has_above and has_below

    if is_bracketed:
        msg = "Value is properly bracketed by comparable sales."
    elif not has_above:
        msg = (
            f"No comparable sale exceeds the appraised value of ${appraised_value:,.0f}. "
            "UAD guidelines recommend at least one comp above the subject value."
        )
    else:
        msg = (
            f"No comparable sale is below the appraised value of ${appraised_value:,.0f}. "
            "UAD guidelines recommend at least one comp below the subject value."
        )

    return BracketingCheck(
        appraised_value=appraised_value,
        comp_prices=valid_prices,
        has_comp_above=has_above,
        has_comp_below=has_below,
        is_bracketed=is_bracketed,
        message=msg,
    )


def _compute_quality_score(
    valid_comp_count: int,
    ppsf_data: dict,
    is_bracketed: bool,
    appraised_in_range: Optional[bool],
) -> tuple[float, str]:
    """
    Score the overall quality of comp selection and value support.

    Factors:
      - Number of valid comps (max 3)
      - Whether subject $/sqft is close to comp mean
      - Whether UAD bracketing is satisfied
      - Whether appraised value falls in adjusted range
    """
    score = 0.0

    # Comp count (0.4 weight)
    score += (valid_comp_count / 3) * 0.4

    # $/sqft proximity (0.3 weight)
    vs_mean = ppsf_data.get("subject_vs_mean_pct")
    if vs_mean is not None:
        proximity = max(0.0, 1.0 - abs(vs_mean) * 2)  # penalty doubles with distance
        score += proximity * 0.3
    else:
        score += 0.15  # partial credit if we couldn't compute

    # Bracketing (0.15 weight)
    score += 0.15 if is_bracketed else 0.0

    # Appraised value in adjusted range (0.15 weight)
    if appraised_in_range is True:
        score += 0.15
    elif appraised_in_range is None:
        score += 0.075  # partial credit

    score = round(min(1.0, max(0.0, score)), 4)

    if score >= 0.75:
        label = "Strong"
    elif score >= 0.50:
        label = "Adequate"
    else:
        label = "Weak"

    return score, label


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------

class CompAdjustmentEngine:
    """
    Analyzes comparable sales data extracted from an appraisal document
    and produces a quantitative assessment of value support.
    """

    def analyze(self, fields: AppraisalFields) -> CompAdjustmentReport:
        """
        Run full comparable sales analysis on extracted appraisal fields.

        Args:
            fields: AppraisalFields from LLMExtractor (already validated).

        Returns:
            CompAdjustmentReport with price analysis, GLA adjustments,
            bracketing check, and quality score.
        """
        warnings: list[str] = []

        subject_value = _parse_currency(fields.appraised_value.value)
        subject_gla = _parse_float(fields.gross_living_area_sqft.value)
        subject_implied_ppsf = (
            round(subject_value / subject_gla, 2)
            if subject_value and subject_gla and subject_gla > 0
            else None
        )

        comps = _parse_comparables(fields)
        valid_comps = [c for c in comps if c.sale_price is not None]

        if len(valid_comps) == 0:
            warnings.append("No valid comparable sales data found. Analysis is limited.")

        # Price-per-sqft analysis
        ppsf_data = _ppsf_analysis(subject_value, subject_gla, comps)

        # GLA adjustment analysis
        gla_rate = _estimate_gla_adjustment_rate(comps)
        gla_adjustments: list[GLAAdjustment] = []
        adj_range_low = adj_range_high = None
        appraised_in_range = None

        if gla_rate and subject_gla:
            gla_adjustments = _compute_gla_adjustments(subject_gla, comps, gla_rate)
            if gla_adjustments:
                adj_values = [a.adjusted_comp_value for a in gla_adjustments]
                adj_range_low = round(min(adj_values), 0)
                adj_range_high = round(max(adj_values), 0)

                if subject_value:
                    appraised_in_range = adj_range_low <= subject_value <= adj_range_high
                    if not appraised_in_range:
                        warnings.append(
                            f"Appraised value (${subject_value:,.0f}) falls outside "
                            f"GLA-adjusted comp range (${adj_range_low:,.0f} – ${adj_range_high:,.0f}). "
                            "This may indicate additional adjustments are needed."
                        )
        else:
            warnings.append(
                "Could not estimate GLA adjustment rate — "
                "need at least 2 comps with both sale price and GLA."
            )

        # Bracketing check
        bracketing = None
        if subject_value and valid_comps:
            bracketing = _check_bracketing(subject_value, comps)
            if not bracketing.is_bracketed:
                warnings.append(bracketing.message)

        # Quality score
        quality_score, quality_label = _compute_quality_score(
            valid_comp_count=len(valid_comps),
            ppsf_data=ppsf_data,
            is_bracketed=bracketing.is_bracketed if bracketing else False,
            appraised_in_range=appraised_in_range,
        )

        # Summary
        summary = self._build_summary(
            subject_value, ppsf_data, quality_label, quality_score,
            bracketing, appraised_in_range, adj_range_low, adj_range_high,
        )

        return CompAdjustmentReport(
            subject_appraised_value=subject_value,
            subject_gla_sqft=subject_gla,
            subject_implied_ppsf=subject_implied_ppsf,
            comparables=comps,
            valid_comp_count=len(valid_comps),
            comp_ppsf_mean=ppsf_data["mean"],
            comp_ppsf_std=ppsf_data["std"],
            comp_ppsf_min=ppsf_data["min"],
            comp_ppsf_max=ppsf_data["max"],
            subject_ppsf_vs_comp_mean_pct=ppsf_data["subject_vs_mean_pct"],
            estimated_gla_rate_per_sqft=gla_rate,
            gla_adjustments=gla_adjustments,
            adjusted_value_range_low=adj_range_low,
            adjusted_value_range_high=adj_range_high,
            appraised_value_in_range=appraised_in_range,
            bracketing=bracketing,
            comp_quality_score=quality_score,
            comp_quality_label=quality_label,
            summary=summary,
            warnings=warnings,
        )

    def _build_summary(
        self,
        subject_value: Optional[float],
        ppsf_data: dict,
        quality_label: str,
        quality_score: float,
        bracketing: Optional[BracketingCheck],
        appraised_in_range: Optional[bool],
        adj_low: Optional[float],
        adj_high: Optional[float],
    ) -> str:
        parts = [f"Comp selection quality: {quality_label} ({quality_score:.0%})."]

        if ppsf_data["mean"]:
            parts.append(
                f"Comp $/sqft range: ${ppsf_data['min']:.0f}–${ppsf_data['max']:.0f} "
                f"(mean ${ppsf_data['mean']:.0f})."
            )

        if subject_value and ppsf_data.get("subject_vs_mean_pct") is not None:
            pct = ppsf_data["subject_vs_mean_pct"]
            direction = "above" if pct > 0 else "below"
            parts.append(
                f"Subject implied $/sqft is {abs(pct):.1%} {direction} comp mean."
            )

        if adj_low and adj_high:
            in_out = "within" if appraised_in_range else "outside"
            parts.append(
                f"Appraised value is {in_out} GLA-adjusted comp range "
                f"(${adj_low:,.0f}–${adj_high:,.0f})."
            )

        if bracketing:
            parts.append(bracketing.message)

        return " ".join(parts)
