# AWS Bedrock RAG Healthcare Summarizer

## Problem Statement

Clinical data in enterprise healthcare systems is fragmented across FHIR JSON schemas and unstructured clinical notes. Clinicians spend excessive time parsing lengthy patient records.

Standard commercial Generative AI implementations present two primary operational risks:

1. **HIPAA and PHI Violations:** Ingesting unsanitized health data into public endpoints violates federal compliance standards.
2. **Hallucination Risks:** Unconstrained LLMs extrapolate or generate clinical facts when context is missing, introducing safety risks into clinical decision workflows.

## Architecture Style

This project implements a secure, source-grounded Retrieval-Augmented Generation (RAG) architecture on AWS.

```
[Raw Clinical Payload / FHIR JSON]
               │
               ▼
   [/guardrails/phi_sanitizer.py] ──► (Redacts PII/PHI)
               │
               ▼
   [/ingestion/fhir_parser.py & chunking_strategy.py]
               │
               ▼
   [/pipeline/bedrock_embeddings.py] ──► (Amazon Titan v2)
               │
               ▼
   [/pipeline/vector_store.py] ──► (OpenSearch Serverless / Pgvector)
               │
               ▼
   [/pipeline/rag_orchestrator.py] ──► (Amazon Bedrock Claude 3)
               │
               ▼
   [/api/main.py] ──► (FastAPI Secure Endpoint)

```
## How to Run Locally

### Prerequisites
* **Python 3.9+** installed
* Active **AWS Account** with permissions and access granted to **AWS Bedrock models** (e.g., Anthropic Claude / Titan Embeddings)
* Configured **AWS CLI** credentials (`aws configure`) or environment variables

### Setup Steps
```bash
# 1. Clone the repository
git clone [https://github.com/bluejaytg/aws-bedrock-rag-healthcare-summarizer.git](https://github.com/bluejaytg/aws-bedrock-rag-healthcare-summarizer.git)
cd aws-bedrock-rag-healthcare-summarizer

# 2. Set up and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env  # Add your AWS_REGION and credentials if not using default AWS CLI profile

# 5. Launch the application
streamlit run app.py  # Or 'python main.py' / 'uvicorn main:app --reload' depending on your entry point

### Core System Principles

* **Input Sanitization:** Ingested text passes through RegEx and NLP scrubbers to redact Protected Health Information (PHI) prior to vector embedding or model inference.
* **Clinical Text Chunking:** Text splitting preserves medical boundaries (clinical observations, section headers) rather than relying on fixed character windows.
* **Strict Context Grounding:** System prompts restrict the model to retrieved context chunks and force a deterministic fallback response (*"Insufficient clinical data provided in context"*) when data is missing.

## Key Observations & Benchmarks

1. **Temperature Tuning:** Temperatures above `0.1` introduced speculative summaries. Setting temperature to `0.0` ensured deterministic, strictly grounded outputs across test payloads.
2. **Chunk Size Optimization:** Small chunk windows (250 tokens) severed context between clinical findings and diagnoses. A window of 512 tokens with 10% overlap preserved clinical context while maximizing vector retrieval precision.
3. **PHI Redaction Overhead:** Local RegEx and pattern matching added approximately 1.2ms per request, representing negligible overhead relative to vector lookup and model invocation latency (~450ms total).

## Repository Structure

```text
aws-bedrock-rag-healthcare-summarizer/
├── /ingestion
│   ├── fhir_parser.py
│   └── chunking_strategy.py
├── /pipeline
│   ├── bedrock_embeddings.py
│   ├── vector_store.py
│   └── rag_orchestrator.py
├── /guardrails
│   ├── phi_sanitizer.py
│   └── source_grounding.py
├── /api
│   └── main.py
├── template.yaml
└── README.md
