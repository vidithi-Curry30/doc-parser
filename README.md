# Appraisal Document Parser

An AI-powered pipeline that extracts structured data from residential property appraisal reports (URAR / UAD 3.6 format) and enables field-by-field comparison across multiple documents.

Built as a demonstration of combining classical document processing with modern LLM-based structured extraction

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────┐
│           PDF Extraction Layer          │
│                                         │
│  Strategy 1: pdfplumber                 │
│    ├─ Table-aware text extraction       │
│    └─ Best for structured URAR forms    │
│                                         │
│  Strategy 2: PyMuPDF (fallback)         │
│    ├─ Broader PDF variant support       │
│    └─ Auto-selected if pdfplumber fails │
└─────────────────────────────────────────┘
    │
    ▼  (raw text + tables per page)
┌─────────────────────────────────────────┐
│          LLM Extraction Layer           │
│                                         │
│  GPT-4o (JSON mode, temp=0)             │
│    ├─ Structured system prompt with     │
│    │   full schema definition           │
│    ├─ 35 fields extracted in one pass   │
│    └─ Per-field confidence scoring      │
│         (high / medium / low / missing) │
└─────────────────────────────────────────┘
    │
    ▼  (raw LLM JSON)
┌─────────────────────────────────────────┐
│       Post-Validation Layer             │
│                                         │
│  Rule-based sanity checks on numeric    │
│  fields:                                │
│    ├─ appraised_value: must be > $1,000 │
│    ├─ year_built: must be 1600–present  │
│    ├─ GLA: 100–50,000 sq ft            │
│    └─ Downgrades confidence on failure  │
└─────────────────────────────────────────┘
    │
    ▼
Pydantic ExtractionResponse
    ├─ 35 AppraisalFields (each with value + confidence + reasoning + source snippet)
    ├─ Confidence summary (counts per level)
    └─ Processing metadata (time, strategy, page count)
```

---

## Features

| Feature | Description |
|---|---|
| **Multi-strategy PDF extraction** | pdfplumber (primary) + PyMuPDF (fallback) with automatic selection |
| **35 appraisal fields** | Property ID, valuation, characteristics, neighborhood, 3 comparable sales |
| **Per-field confidence scoring** | `high` / `medium` / `low` / `missing` with LLM reasoning |
| **Source snippet tracking** | Each field traces back to the exact text it was pulled from |
| **Rule-based post-validation** | Domain sanity checks on numeric fields; downgrades confidence on suspicious values |
| **Document comparison** | Field-by-field diff with numeric delta calculation and LLM narrative summary |
| **REST API** | FastAPI with Swagger docs, typed request/response models |
| **CLI** | Rich terminal output for quick testing without the API server |
| **Full test suite** | Mocked LLM tests covering extraction, validation, and comparison logic |

---

## Extracted Fields

The system extracts all standard UAD 3.6 / URAR form fields:

**Property Identification**
`property_address`, `city`, `state`, `zip_code`, `county`, `legal_description`, `assessors_parcel_number`

**Appraisal Details**
`appraised_value`, `effective_date_of_appraisal`, `appraisal_purpose`, `appraiser_name`, `appraiser_license_number`, `lender_client`

**Property Characteristics**
`gross_living_area_sqft`, `lot_size`, `year_built`, `property_type`, `number_of_bedrooms`, `number_of_bathrooms`, `number_of_stories`, `garage_capacity`, `basement`, `condition_rating`, `quality_rating`

**Market / Neighborhood**
`neighborhood_name`, `market_trend`, `prior_sale_price`, `prior_sale_date`

**Comparable Sales (3)**
`comp_1_address`, `comp_1_sale_price`, `comp_1_gla`, `comp_2_address`, `comp_2_sale_price`, `comp_2_gla`, `comp_3_address`, `comp_3_sale_price`, `comp_3_gla`

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/yourusername/appraisal-doc-parser.git
cd appraisal-doc-parser
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure your OpenAI API key

```bash
cp .env.example .env
# Open .env and replace "your_openai_api_key_here" with your real key
```

Your `.env` file should look like:
```
OPENAI_API_KEY=sk-...your-key-here...
MODEL_NAME=gpt-4o
```

---

## Usage

### Option A: CLI (no server needed)

Extract fields from a single PDF:
```bash
python -m src.cli extract path/to/appraisal.pdf
```

Show only high-confidence fields:
```bash
python -m src.cli extract path/to/appraisal.pdf --high-only
```

Save full JSON output (including confidence and reasoning for every field):
```bash
python -m src.cli extract path/to/appraisal.pdf --output results.json
```

Compare two appraisal PDFs:
```bash
python -m src.cli compare appraisal_v1.pdf appraisal_v2.pdf
```

---

### Option B: REST API

Start the server:
```bash
uvicorn src.api:app --reload
```

The API will be live at `http://localhost:8000`.  
Interactive docs (Swagger UI): `http://localhost:8000/docs`

**Extract a document:**
```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@path/to/appraisal.pdf"
```

**Compare two documents:**
```bash
curl -X POST http://localhost:8000/compare \
  -F "file1=@appraisal_v1.pdf" \
  -F "file2=@appraisal_v2.pdf"
```

**Check which fields are extracted:**
```bash
curl http://localhost:8000/schema
```

---

## Example API Response (`/extract`)

```json
{
  "file_name": "appraisal_123_main_st.pdf",
  "page_count": 8,
  "extraction_strategy": "pdfplumber",
  "total_chars_extracted": 14823,
  "processing_time_seconds": 3.412,
  "confidence_summary": {
    "high": 18,
    "medium": 7,
    "low": 3,
    "missing": 7
  },
  "fields": {
    "appraised_value": {
      "value": "$425,000",
      "confidence": "high",
      "reasoning": "Found directly on page 1 in the reconciliation section.",
      "source_snippet": "Final Reconciliation: $425,000"
    },
    "property_address": {
      "value": "123 Main St",
      "confidence": "high",
      "reasoning": "Clearly stated at the top of the URAR form.",
      "source_snippet": "Subject Property Address: 123 Main St"
    },
    "year_built": {
      "value": "1987",
      "confidence": "high",
      "reasoning": "Stated in property description section.",
      "source_snippet": "Year Built: 1987"
    },
    "condition_rating": {
      "value": "C3",
      "confidence": "medium",
      "reasoning": "Found in condition section, but surrounded by other ratings.",
      "source_snippet": "Condition: C3 - Average"
    }
  }
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests mock the OpenAI API — no API key or network access required.

---

## Project Structure

```
appraisal-doc-parser/
├── src/
│   ├── __init__.py
│   ├── pdf_extractor.py     # Multi-strategy PDF text extraction
│   ├── schemas.py           # Pydantic models (fields, confidence, API responses)
│   ├── llm_extractor.py     # OpenAI GPT-4o extraction + comparison logic
│   ├── api.py               # FastAPI REST endpoints
│   └── cli.py               # Click CLI with Rich terminal output
├── tests/
│   └── test_extractor.py    # Unit tests (mocked LLM)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Design Decisions

**Why two PDF extraction libraries?**
`pdfplumber` excels at structured forms like URAR (table detection, reading order) but fails on some encrypted or non-standard PDFs. `PyMuPDF` handles more edge cases as a fallback, ensuring reliability across different document sources.

**Why temperature=0 for extraction?**
Appraisal field extraction is a deterministic information retrieval task. We want the same document to produce the same output every time — not creative variation. Temperature=0 makes the LLM behave more like a retrieval function.

**Why JSON mode instead of prompting for JSON?**
`response_format={"type": "json_object"}` guarantees parseable output at the API level. Without it, GPT-4o sometimes wraps JSON in markdown code fences, requiring fragile string cleanup.

**Why post-validate numerics with rules?**
LLMs can hallucinate values that are syntactically valid JSON but semantically wrong (e.g., an appraised value of $1 because the document had "$1 million" split across a line break). Rule-based validation catches these domain-specific failure modes cheaply.
