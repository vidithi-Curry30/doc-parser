# Appraisal Document Parser

An end-to-end pipeline for extracting structured data from mortgage loans and residential property appraisal reports (URAR / UAD 3.6 format). Five distinct processing stages — designed to catch failure modes the previous one cannot.

Includes a FastAPI REST API, Click CLI, and async batch processor.

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────┐
│           PDF Extraction Layer          │
│  Strategy 1: pdfplumber (primary)       │
│    └─ Table-aware; optimized for URAR   │
│  Strategy 2: PyMuPDF (fallback)         │
│    └─ Auto-selected on pdfplumber fail  │
└─────────────────────────────────────────┘
    │
    ▼  (raw text per page)
┌─────────────────────────────────────────┐
│          LLM Extraction Layer           │
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
│  Domain sanity checks on 9 numeric      │
│  fields — catches LLM hallucinations    │
│  that are syntactically valid but       │
│  semantically wrong:                    │
│    ├─ appraised_value > $1,000          │
│    ├─ year_built: 1600–present          │
│    └─ GLA: 100–50,000 sq ft            │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│      Cross-Field Calibration Layer      │
│  An independent second opinion on the   │
│  LLM's self-reported confidence scores. │
│  Six domain-specific consistency checks:│
│    ├─ Appraised value vs comp midpoint  │
│    ├─ Subject $/sqft z-score vs comps   │
│    ├─ Year built ↔ UAD condition rating │
│    ├─ Bedroom/bathroom ratio            │
│    ├─ Prior sale vs appraised value     │
│    └─ Critical field completeness rate  │
│  Output: reliability_score (0.0–1.0)    │
│  + per-field calibrated confidence      │
│  + CalibrationFlags (WARNING / ERROR)   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│       Comparable Sales Engine           │
│  Encodes UAD / Fannie Mae methodology:  │
│    ├─ Price-per-sqft stats across comps │
│    ├─ GLA adjustment rate estimation    │
│    ├─ Per-comp GLA-adjusted sale value  │
│    ├─ GLA-adjusted value support range  │
│    ├─ UAD bracketing requirement check  │
│    └─ Comp selection quality score      │
│  Output: comp_quality_score (0–1)       │
│  + adjusted range + bracketing result   │
└─────────────────────────────────────────┘
    │
    ▼
ExtractionResponse (Pydantic)
    ├─ 35 typed AppraisalFields (value + confidence + reasoning + source snippet)
    ├─ CalibrationReport (reliability score + flags + calibrated confidences)
    └─ CompAdjustmentReport ($/sqft stats + GLA adjustments + bracketing + quality score)
```

---

## Features

| Feature | Description |
|---|---|
| **Multi-strategy PDF extraction** | pdfplumber + PyMuPDF fallback, auto-selected per document |
| **35 appraisal fields** | Full URAR/UAD 3.6 coverage: property ID, valuation, characteristics, market context, 3 comps |
| **Per-field provenance** | Every field includes confidence level, LLM reasoning, and verbatim source snippet |
| **Rule-based post-validation** | Sanity checks on 9 numeric fields; downgrades confidence on impossible values |
| **Cross-field calibration** | Six consistency checks produce an independent reliability score (0–1); flags affected fields with WARNING/ERROR severity |
| **Comparable sales engine** | UAD methodology in code: $/sqft analysis, GLA adjustments, adjusted value range, bracketing check, quality score |
| **Document comparison** | Field-by-field diff with numeric deltas and LLM narrative summary |
| **Form 1003 support** | Extracts 28 fields from Uniform Residential Loan Applications |
| **Async batch processing** | asyncio + ThreadPoolExecutor + semaphore; concurrent folder processing with CSV export |
| **REST API + CLI** | FastAPI with Swagger docs; Rich CLI with `--high-only` filter and JSON export |
| **Test suite** | 30+ unit tests — all mocked, no API key required |

---

## Setup

### Option A: Docker (recommended)

```bash
docker build -t appraisal-parser .
docker run -e OPENAI_API_KEY=sk-... -p 8000:8000 appraisal-parser
```

API: `http://localhost:8000` · Swagger: `http://localhost:8000/docs`

### Option B: Local

```bash
git clone https://github.com/yourusername/appraisal-doc-parser.git
cd appraisal-doc-parser
python -m venv env && source env/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY
```

---

## Usage

```bash
# CLI
python -m src.cli extract path/to/appraisal.pdf
python -m src.cli extract path/to/appraisal.pdf --high-only
python -m src.cli extract path/to/appraisal.pdf --output results.json
python -m src.cli compare appraisal_v1.pdf appraisal_v2.pdf

# Batch (concurrent folder processing → CSV)
python -m src.batch_processor ./appraisals/ --output results.csv --concurrency 3

# API
uvicorn src.api:app --reload
curl -X POST http://localhost:8000/extract -F "file=@appraisal.pdf"
curl -X POST http://localhost:8000/compare -F "file1=@v1.pdf" -F "file2=@v2.pdf"
```

---

## Sample Output

Real output from `sample_appraisal.pdf` (included). Full output in `sample_output.json`.

```json
{
  "file_name": "sample_appraisal.pdf",
  "extraction_strategy": "pdfplumber",
  "processing_time_seconds": 23.056,
  "confidence_summary": { "high": 35, "medium": 0, "low": 2, "missing": 0 },
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
      "reasoning": "State label merged with zip due to a document typo.",
      "source_snippet": "State: Zip CIoLde: 62704"
    }
  },
  "calibration": {
    "reliability_score": 0.9091,
    "error_count": 0,
    "warning_count": 0,
    "summary": "Reliability: High (91%). 0 error(s), 0 warning(s). Fields are internally consistent.",
    "flags": []
  },
  "comp_analysis": {
    "subject_appraised_value": 485000.0,
    "subject_implied_ppsf": 225.58,
    "comparables": [
      { "label": "comp_1", "address": "818 Mulberry Lane, Springfield, IL 62704", "sale_price": 472000.0, "gla_sqft": 2080.0, "price_per_sqft": 226.92 },
      { "label": "comp_2", "address": "1204 Oak Street, Springfield, IL 62704",   "sale_price": 498500.0, "gla_sqft": 2210.0, "price_per_sqft": 225.57 },
      { "label": "comp_3", "address": "55 Chestnut Ave, Springfield, IL 62703",    "sale_price": 479000.0, "gla_sqft": 2100.0, "price_per_sqft": 228.10 }
    ],
    "price_per_sqft_analysis": { "comp_mean": 226.86, "comp_std": 1.27, "subject_vs_mean_pct": -0.0057 },
    "gla_adjustment_analysis": {
      "estimated_rate_per_sqft": 56.72,
      "adjusted_value_range": { "low": 475970.0, "high": 495097.0 },
      "appraised_value_in_range": true
    },
    "bracketing": { "is_bracketed": true, "has_comp_above": true, "has_comp_below": true },
    "comp_quality_score": 0.9966,
    "comp_quality_label": "Strong",
    "summary": "Comp $/sqft range: $226–$228 (mean $227). Subject implied $/sqft within 0.6% of comp mean. Appraised value within GLA-adjusted range ($475,970–$495,097). UAD bracketing satisfied."
  }
}
```

**35 of 37 fields at high confidence.** The 2 low-confidence fields trace directly to a text corruption in the source (`State: Zip CIoLde: 62704`) — the system flags them rather than hallucinating a value. Calibration independently verified internal consistency (reliability **0.91**, 0 flags). Comp engine confirmed appraised value is within the GLA-adjusted range and UAD bracketing is satisfied. Comp quality: **Strong (99.7%)**.

---

## Running Tests

```bash
pytest tests/ -v
```

All tests mock the OpenAI API — no key or network required.

---

## Project Structure

```
appraisal-doc-parser/
├── src/
│   ├── schemas.py                # Pydantic models: ExtractedField, AppraisalFields, LoanFields, response types
│   ├── pdf_extractor.py          # Dual-strategy PDF extraction (pdfplumber + PyMuPDF)
│   ├── llm_extractor.py          # GPT-4o extraction, post-validation, document comparison
│   ├── confidence_calibrator.py  # Cross-field consistency checks, reliability scoring, flag generation
│   ├── comp_engine.py            # Comparable sales engine: $/sqft, GLA adjustments, UAD bracketing
│   ├── loan_extractor.py         # Form 1003 (URLA) extraction — 28 loan application fields
│   ├── batch_processor.py        # Async batch pipeline: asyncio + ThreadPoolExecutor + semaphore, CSV export
│   ├── api.py                    # FastAPI: /extract, /compare, /health, /schema
│   └── cli.py                    # Click CLI with Rich terminal output
├── tests/
│   ├── test_extractor.py
│   └── test_calibration_and_comps.py
├── sample_appraisal.pdf
├── sample_output.json
├── Dockerfile
├── .env.example
├── requirements.txt
└── README.md
```

---

## Design Decisions

**Why a separate calibration layer instead of trusting LLM confidence scores?**
LLM self-reported confidence is uncalibrated — the model can assign `"high"` to a hallucinated value. The calibration layer asks a different question: *do the extracted fields agree with each other?* An appraised value 80% above the comp midpoint is a data quality signal regardless of what the LLM reported. Detected inconsistencies propagate back as confidence downgrades on the affected fields, with `severity`, `affected_fields`, `message`, and `suggested_action` recorded per flag.

**Why implement comp analysis in code rather than prompting the LLM?**
UAD appraisal methodology — bracketing, GLA adjustments, $/sqft analysis — is quantitative and deterministic. Code is more reliable, auditable, and testable than LLM arithmetic. It also runs entirely on already-extracted fields, adding zero API cost or latency.

**Why rule-based numeric validation between LLM extraction and calibration?**
LLMs can return values that are syntactically valid JSON but semantically impossible — `$1` appraised value from `"$1 million"` spanning a line break, or `year_built: 2090` from an OCR artifact. Rules catch these cheaply before they reach the calibrator and corrupt its consistency checks.
