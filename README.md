# Appraisal Document Parser

An end-to-end pipeline for extracting structured data from residential property appraisal reports (URAR / UAD 3.6 format). Built with five distinct processing stages — PDF extraction, LLM extraction, rule-based validation, cross-field calibration, and automated comparable sales analysis — each designed to catch failure modes the previous stage cannot.

Includes a FastAPI REST API, Click CLI, and an async batch processor for folder-scale processing.

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────┐
│           PDF Extraction Layer          │
│                                         │
│  Strategy 1: pdfplumber (primary)       │
│    └─ Table-aware; optimized for URAR   │
│                                         │
│  Strategy 2: PyMuPDF (fallback)         │
│    └─ Broader variant support;          │
│       auto-selected on pdfplumber fail  │
└─────────────────────────────────────────┘
    │
    ▼  (raw text per page)
┌─────────────────────────────────────────┐
│          LLM Extraction Layer           │
│                                         │
│  GPT-4o (JSON mode, temperature=0)      │
│    ├─ Full schema definition in prompt  │
│    ├─ 35 fields extracted in one pass   │
│    └─ Per-field confidence + reasoning  │
│       + verbatim source snippet         │
└─────────────────────────────────────────┘
    │
    ▼  (raw LLM JSON)
┌─────────────────────────────────────────┐
│       Rule-Based Post-Validation        │
│                                         │
│  Domain sanity checks on 9 numeric      │
│  fields — catches LLM hallucinations    │
│  that are syntactically valid but       │
│  semantically wrong:                    │
│    ├─ appraised_value > $1,000          │
│    ├─ year_built: 1600–present          │
│    └─ GLA: 100–50,000 sq ft            │
│  Confidence is downgraded on failure    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│      Cross-Field Calibration Layer      │
│                                         │
│  An independent second opinion on the   │
│  LLM's self-reported confidence scores. │
│  Checks internal consistency across     │
│  field pairs using six domain rules:    │
│                                         │
│    ├─ Appraised value vs comp midpoint  │
│    ├─ Subject $/sqft z-score vs comps   │
│    ├─ Year built ↔ UAD condition rating │
│    ├─ Bedroom/bathroom ratio            │
│    ├─ Prior sale vs appraised value     │
│    └─ Critical field completeness rate  │
│                                         │
│  Output: reliability_score (0.0–1.0)    │
│  + per-field calibrated confidence      │
│  + CalibrationFlags (WARNING / ERROR)   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│       Comparable Sales Engine           │
│                                         │
│  Encodes standard appraisal methodology │
│  (UAD / Fannie Mae guidelines):         │
│                                         │
│    ├─ Price-per-sqft stats across comps │
│    ├─ GLA adjustment rate estimation    │
│    ├─ Per-comp GLA-adjusted sale value  │
│    ├─ GLA-adjusted value support range  │
│    ├─ UAD bracketing requirement check  │
│    └─ Comp selection quality score      │
│                                         │
│  Output: comp_quality_score (0–1)       │
│  + adjusted range + bracketing result   │
└─────────────────────────────────────────┘
    │
    ▼
ExtractionResponse (Pydantic)
    ├─ 35 typed AppraisalFields (value + confidence + reasoning + source snippet)
    ├─ Confidence summary (field counts by level)
    ├─ CalibrationReport (reliability score + flags + calibrated confidences)
    └─ CompAdjustmentReport ($/sqft stats + GLA adjustments + bracketing + quality score)
```

---

## Features

| Feature | Description |
|---|---|
| **Multi-strategy PDF extraction** | pdfplumber (primary, table-aware) + PyMuPDF (fallback), automatically selected per document; strategy is logged in every response |
| **35 appraisal fields** | Full URAR/UAD 3.6 coverage: property identification, valuation, physical characteristics, market context, and 3 comparable sales |
| **Per-field provenance** | Every extracted value includes its confidence level, LLM reasoning trace, and verbatim source snippet from the document text |
| **Rule-based post-validation** | Domain sanity checks on 9 numeric fields; catches values that are syntactically valid JSON but semantically impossible for appraisal data |
| **Cross-field calibration** | Six domain-specific consistency checks produce an independent reliability score (0–1) that is separate from — and can override — the LLM's self-reported confidence; inconsistent field pairs are flagged with severity (WARNING / ERROR) and affected fields are downgraded |
| **Comparable sales engine** | Quantitative implementation of UAD appraisal methodology: $/sqft statistics, GLA adjustment rate estimation, per-comp adjusted values, adjusted value range, Fannie Mae bracketing check, and an overall comp quality score |
| **Document comparison** | Field-by-field diff with numeric delta calculation and LLM-generated narrative summary of key discrepancies |
| **Form 1003 support** | Extracts 28 fields from Uniform Residential Loan Applications (URLA) alongside appraisal documents |
| **Async batch processing** | asyncio + ThreadPoolExecutor + semaphore rate limiting; processes a folder of PDFs concurrently with real-time Rich progress UI; exports a flat CSV with value, confidence, reliability score, and comp quality per document |
| **REST API** | FastAPI with full typed request/response models, Swagger docs, liveness probe (`/health`), and schema introspection (`/schema`) |
| **CLI** | Rich terminal output with `--high-only` confidence filter and `--output` JSON export |
| **Test suite** | 30+ unit tests for calibrator, comp engine, and extraction logic — all mocked; no API key or network required |

---

## Extracted Fields

**Property Identification**
`property_address` · `city` · `state` · `zip_code` · `county` · `legal_description` · `assessors_parcel_number`

**Appraisal Details**
`appraised_value` · `effective_date_of_appraisal` · `appraisal_purpose` · `appraiser_name` · `appraiser_license_number` · `lender_client`

**Property Characteristics**
`gross_living_area_sqft` · `lot_size` · `year_built` · `property_type` · `number_of_bedrooms` · `number_of_bathrooms` · `number_of_stories` · `garage_capacity` · `basement` · `condition_rating` · `quality_rating`

**Market / Neighborhood**
`neighborhood_name` · `market_trend` · `prior_sale_price` · `prior_sale_date`

**Comparable Sales (3 comps × 3 fields)**
`comp_1_address` · `comp_1_sale_price` · `comp_1_gla` · `comp_2_address` · `comp_2_sale_price` · `comp_2_gla` · `comp_3_address` · `comp_3_sale_price` · `comp_3_gla`

---

## Setup

### Option A: Docker (recommended)

```bash
docker build -t appraisal-parser .
docker run -e OPENAI_API_KEY=sk-... -p 8000:8000 appraisal-parser
```

API live at `http://localhost:8000` · Swagger UI at `http://localhost:8000/docs`

### Option B: Local

```bash
git clone https://github.com/yourusername/appraisal-doc-parser.git
cd appraisal-doc-parser
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env:
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-4o
```

---

## Usage

### CLI

```bash
# Extract a single PDF (all fields + calibration + comp analysis)
python -m src.cli extract path/to/appraisal.pdf

# Show high-confidence fields only
python -m src.cli extract path/to/appraisal.pdf --high-only

# Export full JSON output
python -m src.cli extract path/to/appraisal.pdf --output results.json

# Compare two appraisals field by field
python -m src.cli compare appraisal_v1.pdf appraisal_v2.pdf
```

### Batch Processing

```bash
python -m src.batch_processor ./appraisals/ --output results.csv --concurrency 3
```

Processes all PDFs in the folder concurrently. The output CSV has one row per document with columns for every extracted field value, its confidence level, the document's overall reliability score, calibration flag counts, and the comparable sales quality score. Useful for portfolio-level analysis.

### REST API

```bash
uvicorn src.api:app --reload
# API:       http://localhost:8000
# Swagger:   http://localhost:8000/docs
```

```bash
# Extract a document
curl -X POST http://localhost:8000/extract \
  -F "file=@path/to/appraisal.pdf"

# Compare two documents
curl -X POST http://localhost:8000/compare \
  -F "file1=@appraisal_v1.pdf" \
  -F "file2=@appraisal_v2.pdf"

# Inspect the field schema
curl http://localhost:8000/schema
```

---

## Sample Output

The following is real output from `sample_appraisal.pdf` (included in this repo), generated by a single call to `/extract`. The full output is in `sample_output.json`.

```json
{
  "file_name": "sample_appraisal.pdf",
  "page_count": 1,
  "extraction_strategy": "pdfplumber",
  "processing_time_seconds": 23.056,
  "confidence_summary": {
    "high": 35,
    "medium": 0,
    "low": 2,
    "missing": 0
  },
  "fields": {
    "appraised_value": {
      "value": "$485,000",
      "confidence": "high",
      "reasoning": "Clearly stated under the 'APPRAISAL DETAILS' section.",
      "source_snippet": "Appraised Value: $485,000"
    },
    "state": {
      "value": null,
      "confidence": "low",
      "reasoning": "Not clearly visible — state label merged with zip due to a document typo.",
      "source_snippet": "State: Zip CIoLde: 62704"
    },
    "condition_rating": {
      "value": "C3",
      "confidence": "high",
      "reasoning": "Clearly stated under the 'PROPERTY CHARACTERISTICS' section.",
      "source_snippet": "Condition Rating: C3"
    }
  },
  "calibration": {
    "reliability_score": 0.9091,
    "error_count": 0,
    "warning_count": 0,
    "critical_fields_present": 0.9091,
    "summary": "Reliability: High (91%). Critical field completeness: 91%. 0 error(s), 0 warning(s). Fields are internally consistent. Extraction is reliable.",
    "flags": []
  },
  "comp_analysis": {
    "subject_appraised_value": 485000.0,
    "subject_gla_sqft": 2150.0,
    "subject_implied_ppsf": 225.58,
    "valid_comp_count": 3,
    "comparables": [
      { "label": "comp_1", "address": "818 Mulberry Lane, Springfield, IL 62704", "sale_price": 472000.0, "gla_sqft": 2080.0, "price_per_sqft": 226.92 },
      { "label": "comp_2", "address": "1204 Oak Street, Springfield, IL 62704",   "sale_price": 498500.0, "gla_sqft": 2210.0, "price_per_sqft": 225.57 },
      { "label": "comp_3", "address": "55 Chestnut Ave, Springfield, IL 62703",    "sale_price": 479000.0, "gla_sqft": 2100.0, "price_per_sqft": 228.10 }
    ],
    "price_per_sqft_analysis": {
      "comp_mean": 226.86,
      "comp_std": 1.27,
      "comp_min": 225.57,
      "comp_max": 228.10,
      "subject_vs_mean_pct": -0.0057
    },
    "gla_adjustment_analysis": {
      "estimated_rate_per_sqft": 56.72,
      "adjusted_value_range": { "low": 475970.0, "high": 495097.0 },
      "appraised_value_in_range": true,
      "adjustments": [
        { "comp": "comp_1", "gla_difference_sqft":  70.0, "adjustment":  3970.0, "adjusted_comp_value": 475970.0 },
        { "comp": "comp_2", "gla_difference_sqft": -60.0, "adjustment": -3403.0, "adjusted_comp_value": 495097.0 },
        { "comp": "comp_3", "gla_difference_sqft":  50.0, "adjustment":  2836.0, "adjusted_comp_value": 481836.0 }
      ]
    },
    "bracketing": {
      "is_bracketed": true,
      "has_comp_above": true,
      "has_comp_below": true,
      "message": "Value is properly bracketed by comparable sales."
    },
    "comp_quality_score": 0.9966,
    "comp_quality_label": "Strong",
    "summary": "Comp selection quality: Strong (100%). Comp $/sqft range: $226–$228 (mean $227). Subject implied $/sqft is 0.6% below comp mean. Appraised value is within GLA-adjusted comp range ($475,970–$495,097). Value is properly bracketed by comparable sales."
  }
}
```

**Reading the output:** 35 of 37 fields extracted at high confidence; the 2 low-confidence fields correspond to a text corruption in the source document (`State: Zip CIoLde: 62704`) — the system correctly identifies and flags them rather than hallucinating a value. The calibration engine independently verified that all extracted values are internally consistent (reliability score: **0.91**, 0 errors). The comp engine confirmed the appraised value ($485,000) falls within the GLA-adjusted comp range ($475,970–$495,097), the subject's implied $/sqft ($225.58) is within 0.6% of the comp mean ($226.86), and UAD bracketing is satisfied. Comp quality: **Strong (99.7%)**.

---

## Running Tests

```bash
pytest tests/ -v
```

All tests mock the OpenAI API — no API key or network access required. Coverage includes:
- Calibrator: value-vs-comp deviation, $/sqft z-score, year-built/condition mismatch, bedroom/bathroom ratio, prior sale jump, critical field completeness
- Comp engine: bracketing logic, GLA adjustment rate estimation, unbounded comp handling, serialization
- Extraction: rule-based numeric validation, Pydantic model construction, confidence downgrade logic

---

## Project Structure

```
appraisal-doc-parser/
├── src/
│   ├── schemas.py                # Pydantic models: ExtractedField, AppraisalFields, LoanFields, all API response types
│   ├── pdf_extractor.py          # Dual-strategy PDF text extraction (pdfplumber + PyMuPDF)
│   ├── llm_extractor.py          # GPT-4o extraction, rule-based post-validation, document comparison
│   ├── confidence_calibrator.py  # Cross-field consistency checks, reliability scoring, flag generation
│   ├── comp_engine.py            # Comparable sales engine: $/sqft analysis, GLA adjustments, UAD bracketing
│   ├── loan_extractor.py         # Form 1003 (URLA) extraction — 28 loan application fields
│   ├── batch_processor.py        # Async batch pipeline: asyncio + ThreadPoolExecutor + semaphore, CSV export
│   ├── api.py                    # FastAPI: /extract, /compare, /health, /schema
│   └── cli.py                    # Click CLI with Rich terminal output
├── tests/
│   ├── test_extractor.py              # Extraction pipeline tests (mocked OpenAI)
│   └── test_calibration_and_comps.py  # 30+ unit tests for calibrator and comp engine (no API calls)
├── sample_appraisal.pdf
├── sample_output.json
├── Dockerfile
├── .env.example
├── requirements.txt
└── README.md
```

---

## Design Decisions

**Why two PDF extraction strategies?**
`pdfplumber` excels at structured forms like URAR (table detection, reading order) but fails on some encrypted or non-standard PDFs. `PyMuPDF` handles more edge cases as a fallback. The strategy used is included in every API response so extraction quality is traceable per document.

**Why temperature=0?**
Appraisal extraction is deterministic retrieval, not generation. The same document should produce the same output every time — this also makes failures reproducible and debuggable.

**Why JSON mode instead of prompting for JSON?**
`response_format={"type": "json_object"}` guarantees parseable output at the API level. Without it, GPT-4o occasionally wraps JSON in markdown code fences, which requires fragile post-processing string cleanup.

**Why rule-based numeric validation after LLM extraction?**
LLMs can return values that are syntactically valid JSON but semantically wrong — an appraised value of `$1` from a document where `"$1 million"` spanned a line break, or a `year_built` of `2090` from an OCR artifact. Rules catch these domain-specific failure modes cheaply, before they propagate downstream.

**Why a separate calibration layer instead of trusting LLM confidence scores?**
LLM self-reported confidence is uncalibrated — the model can assign `"high"` confidence to a value it hallucinated. The calibration layer adds an independent signal by asking a different question: *do the extracted fields agree with each other?* If the appraised value is 80% above the comp midpoint, that deviation is a data quality signal regardless of what confidence the LLM assigned. When inconsistencies are detected, the calibrator downgrades the confidence of the affected fields and records a flag with `severity`, `affected_fields`, `message`, and `suggested_action`.

**Why implement comp analysis in code rather than prompting the LLM for it?**
UAD appraisal methodology — bracketing, GLA adjustments, $/sqft analysis — is quantitative and deterministic. Encoding it in code is more reliable, auditable, and testable than asking an LLM to perform the arithmetic. It also runs entirely on already-extracted fields, adding zero additional API cost or latency.

**Why asyncio + ThreadPoolExecutor for batch processing?**
The OpenAI SDK and PDF libraries are synchronous. Running them directly in asyncio would block the event loop. `run_in_executor` bridges them: the event loop stays non-blocking, `asyncio.Semaphore` caps concurrent API calls to stay within rate limits, and `asyncio.as_completed` means the Rich progress bar updates in real time as each file finishes — not all at once at the end.
