# TruthLens AI

**Autonomous Evidence Retrieval, Neural Claim Verification & Factual Synthesis Platform**

---

## Table of Contents
1. [Overview](#overview)
2. [Core Architectural Law](#core-architectural-law)
3. [System Architecture](#system-architecture)
4. [Key Features](#key-features)
5. [Machine Learning Engine](#machine-learning-engine)
   - [Model A: Evidence Relevance](#model-a-evidence-relevance)
   - [Model B: Claim Verification](#model-b-claim-verification)
   - [Hardware Acceleration & Telemetry](#hardware-acceleration--telemetry)
6. [Automated Evidence Retrieval & Scraping](#automated-evidence-retrieval--scraping)
7. [LLM Explanation Layer](#llm-explanation-layer)
8. [Frontend Design & Typography](#frontend-design--typography)
9. [Production Security Perimeter](#production-security-perimeter)
10. [REST API Specification](#rest-api-specification)
11. [Installation & Setup](#installation--setup)
12. [Environment Configuration](#environment-configuration)
13. [Running Locally](#running-locally)
14. [Docker Deployment](#docker-deployment)
15. [Automated Testing & Benchmarks](#automated-testing--benchmarks)
16. [Repository Structure](#repository-structure)
17. [License](#license)

---

## Overview

**TruthLens AI** is an autonomous news research and claim verification platform engineered to detect and analyze misinformation through a decoupled, multi-stage artificial intelligence architecture.

Traditional AI fact-checking tools often delegate veracity judgments entirely to generative large language models (LLMs). This introduces critical vulnerabilities: generative hallucinations, outdated parametric memory, non-deterministic outputs, and susceptibility to adversarial prompt injections.

TruthLens AI eliminates these vulnerabilities through a strict division of responsibility:
- **Neural transformers** trained on empirical claim-evidence benchmarks determine the mathematical veracity verdict.
- **Generative LLMs** are restricted to translating verified evidence passages and model telemetry into an objective, citation-grounded narrative dossier.

---

## Core Architectural Law

> ### **The ML models determine the verdict. The LLM generates the explanation.**
> 
> The fine-tuned neural models (Model A & Model B) are the sole authorities on truthfulness. The LLM acts strictly as an objective explainer that receives the fixed verdict and verified evidence passages within a hardened prompt boundary (`<UNTRUSTED_EVIDENCE>`). Under no circumstance can the LLM override, alter, or invert the mathematical verdict produced by the neural verification engine.

---

## System Architecture

TruthLens AI decouples web discovery, passage chunking, deep learning inference, data caching, and user presentation across specialized services:

```
[ User Claim Input (Browser) ]
              │
              ▼
    React 18 SPA (Vite :3000)
    ├── Humanized Editorial Theme (Cream, White, Black, Truth-Green, False-Red)
    ├── Times New Roman & Tinos Webfont Typography
    └── Interactive Dossiers, Softmax Distributions, Real-time Status Board
              │
              │  HTTP POST /api/research
              ▼
    Node.js + Express Backend (:5000)
    ├── Security Perimeter (Helmet, CORS Whitelist, IP Rate Limiter, Body Clamp)
    ├── Input Sanitization & Boundary Validation
    ├── Multi-Query Search Generator (DuckDuckGo, Google News, Wikipedia)
    ├── SSRF-Safe Web Scraper & Cheerio DOM Density Parser
    └── Sentence-Boundary Passage Chunker (Abbreviation & Decimal Safe)
              │
              │  HTTP POST /predict
              ▼
    Python FastAPI ML Service (:8000)
    ├── Model A: RoBERTa Evidence Relevance (CUDA GPU / CPU Fallback)
    │     ↳ Filters candidate passages (Decision Threshold P ≥ 0.25 (Calibrated))
    │     ↳ Soft entity-overlap fallback tier prevents premature early exit
    │
    └── Model B: SciFact Claim Verification (CUDA GPU / CPU Fallback)
          ↳ Evaluates claim-evidence entailment: SUPPORTS / REFUTES / NOT_ENOUGH_INFO
          ↳ Relevance-weighted probability pooling across verified evidence
          ↳ Returns AUTHORITATIVE VERDICT & Softmax Distribution
              │
              │  Fixed Verdict + Top Passages
              ▼
    LLM Explanation Layer (Node.js)
    ├── Prompt Boundary Isolation (<UNTRUSTED_EVIDENCE>)
    ├── Multi-Provider Gateway (OpenAI gpt-4o-mini / Google Gemini 1.5)
    ├── Verdict Consistency Guard (discards any contradictory generative text)
    └── Fallback Handler (operates 100% offline if LLM key is absent)
              │
              │  Persist Record & Cache
              ▼
    MongoDB Atlas Cloud Cluster
    ├── Indexed FactCheck Collection
    └── 2-Hour TTL Cache (Sub-15ms query resolution)
```

---

## Key Features

- **Automated Evidence Discovery**: Generates targeted multi-term search queries across general web and news sources, automatically pulling articles and Wikipedia encyclopedic entries.
- **SSRF Ingress Defense**: Validates all outbound scraping requests against 17 attack vectors (blocking loopbacks, private CIDR blocks, cloud metadata endpoints, and non-HTML binaries).
- **Dual-Stage Transformer Pipeline**:
  - **Model A (Relevance)**: Neural sequence classifier filters out retrieval noise.
  - **Model B (Verification)**: Entailment transformer establishes authoritative veracity.
- **Strict Verdict Supremacy**: The LLM cannot change the verdict. Contradictory explanations are automatically rejected by consistency checkers.
- **Prompt Injection Defense**: Retrieved web text is isolated within `<UNTRUSTED_EVIDENCE>` tags, preventing malicious web pages from hijacking instructions.
- **Sub-15ms Caching**: MongoDB compound indexing with a 2-hour TTL cache serves identical claims instantly.
- **Humanized Editorial UI**: Built with Tailwind CSS v4, featuring a print-journalism aesthetic (warm cream canvas, paper cards, charcoal ink, and Times New Roman typography).
- **Full Operational Traceability**: Every dossier details primary reporting publishers, passage relevance scores, softmax probabilities, and GPU inference timings.

---


### Phase 12: Retrieval Quality & NOT_ENOUGH_INFO Calibration
In Phase 12, the retrieval and verification pipeline was upgraded to eliminate false `NOT_ENOUGH_INFO` outcomes while preserving honest uncertainty:
- **Multi-Query Strategy**: Every claim generates up to 4 bounded search queries (exact claim, entity keywords, reference lookup, and factual evidence query).
- **Source Prioritization**: 5-tier source scoring prioritizing `.edu`, `.gov`, peer-reviewed scientific journals, Wikipedia, and verified wire services while strictly blocking clickbait tabloids, forums, and social media.
- **Hardened Extraction**: Cleans Wikipedia and news body text, strips reference markers, and caps extraction at 15,000 words / 5MB with full SSRF defenses.
- **Context-Preserving Chunking**: 1-sentence overlap between consecutive passages prevents broken antecedents and preserves research context.
- **General Domain Generalization**: Calibrated Model A's biomedical-trained weights with high-fidelity encyclopedic entity overlap (>= 80%), achieving 90.9% Model B accuracy across science, technology, history, and geography.
- **Controlled Retrieval Retry**: Automatically triggers a query reformulation retry if Attempt 1 produces zero relevant evidence within a 25-second budget.
- **Relevance-Weighted Aggregation**: Aggregates Model B probabilities weighted by Model A relevance scores with automated contradiction detection.

## Machine Learning Engine

### Model A: Evidence Relevance
- **Architecture**: `roberta-base` Sequence Classification (Binary)
- **Task**: Determine whether an extracted web passage is semantically relevant to the claim.
- **Input Format**: `<s> [CLAIM] claim text </s></s> candidate evidence text </s>`
- **Output Classes**: `0` (`NOT_RELEVANT`), `1` (`RELEVANT`)
- **Calibrated Threshold**: `0.25` (empirically calibrated; configurable via `EVIDENCE_RELEVANCE_THRESHOLD`)
- **Test Set Accuracy**: **90.35%** (+7.15% over TF-IDF baseline)
- **Macro F1 Score**: **78.41%** | **Weighted F1 Score**: **90.46%**
- **ROC-AUC**: **0.9032** | **PR-AUC**: **0.6534**
- **Throughput**: **150.04 samples / second** on NVIDIA RTX 2050
- **Early Exit**: If zero passages meet the relevance criteria, the pipeline terminates early with `NOT_ENOUGH_INFO`, conserving GPU compute and preventing hallucination.

### Model B: Claim Verification
- **Architecture**: `roberta-base` Sequence Classification (Fine-Tuned on SciFact)
- **Task**: Authoritative entailment classification of claim-evidence pairs.
- **Output Classes**:
  - **`SUPPORTS`**: Evidence confirms the claim.
  - **`REFUTES`**: Evidence contradicts or disproves the claim.
  - **`NOT_ENOUGH_INFO`**: Evidence is insufficient or inconclusive.
- **Validation Accuracy**: **78.57%** (SciFact benchmark, n = 448)
- **Macro F1 Score**: **79.00%** | **Weighted F1 Score**: **78.31%**
- **Macro Precision**: **79.38%** | **Macro Recall**: **78.76%**
- **Class Breakdown**:
  - `SUPPORTS`: F1 **78.38%** (Precision: 76.32%, Recall: 80.56%)
  - `REFUTES`: F1 **58.62%** (Precision: 61.82%, Recall: 55.74%)
  - `NOT_ENOUGH_INFO`: F1 **100.00%** (Precision: 100.00%, Recall: 100.00%)
- **Aggregation Strategy**: Relevance-weighted probability pooling across all verified passages determines the final consensus distribution and verdict.

### Hardware Acceleration & Telemetry
- **Primary GPU**: NVIDIA GeForce RTX 2050 (4.0 GB dedicated VRAM)
- **Compute Stack**: CUDA 12.4 + cuDNN with PyTorch 2.6.0 FP16 mixed precision
- **Memory Footprint**: Models A and B remain resident in VRAM (~953 MB combined footprint)
- **Automatic CPU Fallback**: If an NVIDIA GPU is not detected, the service automatically initializes models on the CPU without throwing fatal errors.

---

## Automated Evidence Retrieval & Scraping

1. **Entity-Aware Multi-Query Generation**: Extracted keywords and entities from the claim form search queries dispatched to DuckDuckGo, Google News, and Wikipedia MediaWiki APIs.
2. **SSRF-Safe Web Crawler**:
   - Blocks loopbacks (`localhost`, `127.0.0.1`, `::1`)
   - Blocks private IPv4 subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `100.64.0.0/10`)
   - Blocks cloud metadata endpoints (`169.254.169.254`, `metadata.google.internal`)
   - Blocks binary files (`.exe`, `.pdf`, `.zip`, `.bin`, `.iso`)
   - Rejects decimal/hexadecimal IP obfuscation
3. **Cheerio DOM Extraction**: Scrapes body containers prioritizing highest paragraph density (`maxPCount >= 3`) while stripping scripts, navigation chrome, ads, and footers.
4. **Abbreviation & Decimal-Safe Chunking**: Sentence splitting algorithm preserves numeric decimals (`0 °C`, `2.04 million`) and abbreviations (`e.g.`, `i.e.`, `Dr.`, `Fig.`), preventing broken sentence fragments.

---

## LLM Explanation Layer

The LLM operates strictly downstream of Model B:
- **Supported Providers**: OpenAI (`gpt-4o-mini`, default) and Google Gemini (`gemini-1.5-flash`).
- **Low-Temperature Sampling**: Configured at T = 0.2 to minimize creativity and maximize factual adherence.
- **Strict Prompt Boundary**: External retrieved passages are wrapped within `<UNTRUSTED_EVIDENCE>` XML-like blocks to prevent prompt injection.
- **Structured Schema**:
  - `summary`: 1–2 sentence executive summary.
  - `verdictExplanation`: Evidence-grounded rationale for the verdict.
  - `keyEvidence`: Direct factual citations referencing source publishers.
  - `contradictoryEvidence`: Documented counter-claims or dissenting findings.
  - `uncertainty`: Explicit assessment of evidentiary gaps or limitations.
- **Verdict Consistency Guard**: If the generated text contradicts the neural model verdict, the system retries with an explicit alignment instruction. If it still conflicts, the explanation is dropped and the model verdict is presented cleanly.
- **Graceful Fallback**: If LLM API keys are unconfigured or fail, the fact check completes successfully with raw evidence passages and ML distributions.

---

## Frontend Design & Typography

The frontend is built with **Tailwind CSS v4** to deliver an authentic, humanized, investigative journalism experience:

### Color Palette
- **Warm Cream Canvas**: `#faf7f0` (`--color-cream-100`) and `#f4efe6` (`--color-cream-200`)
- **Crisp Paper White**: `#ffffff` (`--color-paper`) with soft `#ebe4d8` borders
- **Charcoal Black Ink**: `#1c1d22` (`--color-ink-900`) and `#121316` (`--color-ink-950`)
- **Forest Truth Green**: `#15803d` / `#14532d` on `#f0fdf4` (`--color-truth-green-bg`) for `SUPPORTS`
- **Crimson Red**: `#b91c1c` / `#7f1d1d` on `#fef2f2` (`--color-false-red-bg`) for `REFUTES`
- **Warm Amber**: `#b45309` / `#78350f` on `#fffbeb` (`--color-amber-neutral-bg`) for `NOT_ENOUGH_INFO`

### Typography
- **Primary Body & Headings**: **Times New Roman** (`'Times New Roman', 'Tinos', Times, 'Liberation Serif', serif`) with Google Fonts `Tinos` fallback for universal cross-platform rendering.
- **Technical Telemetry**: **JetBrains Mono** (`'JetBrains Mono', monospace`) for confidence percentages, latency numbers, and model parameters.

### Page Routes
- **`/` (Research Page)**: Clean editorial masthead, claim entry workbench with quick-inquiry chips, real-time pipeline status board, verdict stamp, probability distribution, and evidence dossiers.
- **`/report/:id` (Report Page)**: Printable investigative dossier with executive synthesis, categorized evidence, and source authority index.
- **`/history` (History Page)**: Archival ledger with instant client search, verdict filtering tabs, and pagination.
- **`/manual` (Manual Page)**: Laboratory workbench to evaluate claims against user-provided candidate passages directly.
- **`/architecture` (Architecture Page)**: Technical specification, ASCII pipeline diagram, model benchmark tables, and Phase 10 security breakdown.

---

## Production Security Perimeter

- **HTTP Headers**: Enforced via Helmet (HSTS, X-Content-Type-Options, Frameguard).
- **CORS Whitelisting**: Strict origin validation against `CORS_ORIGIN`.
- **IP Rate Limiting**:
  - Research endpoint: 30 requests / 15 minutes / IP.
  - General endpoints: 120 requests / 15 minutes / IP.
  - Health checks: Unthrottled.
- **Body & Timeout Clamps**: Request body clamped at 1MB; request timeout set to 60 seconds.
- **Input Validation**: Centralized boundary middleware returns clean 400 Bad Request responses for invalid payloads.
- **Database Hardening**: Strict 24-hex ObjectId validation, compound indexes `{ verdict: 1, createdAt: -1 }`, and pagination limits (1 <= limit <= 100).
- **Client Hygiene**: Zero `dangerouslySetInnerHTML` usages, `rel="noopener noreferrer"` on all external links, and global React `ErrorBoundary`.

---

## REST API Specification

**Base URL**: `http://localhost:5000/api`

### Endpoints Overview

| Method | Path | Description | Rate Limit |
|---|---|---|---|
| `POST` | `/research` | Automated Fact Check (Web crawl + ML verification + LLM synthesis) | 30 req / 15m |
| `GET` | `/research/:id/report` | Retrieve printable investigative dossier by ID | 120 req / 15m |
| `POST` | `/fact-check` | Manual Fact Check (Verify claim against custom passages) | 120 req / 15m |
| `GET` | `/fact-checks` | Retrieve paginated fact-check history (`?page=1&limit=10`) | 120 req / 15m |
| `GET` | `/fact-checks/:id` | Retrieve single fact-check record by ObjectId | 120 req / 15m |
| `DELETE`| `/fact-checks/:id` | Delete saved fact-check record | 120 req / 15m |
| `GET` | `/health` | Node.js, MongoDB, uptime, and memory RSS telemetry | Unthrottled |
| `GET` | `/health/ml` | FastAPI ML service reachability, CUDA device status, and loaded models | Unthrottled |

### Sample Request: `POST /api/research`
```json
{
  "claim": "Smoking causes lung cancer.",
  "refresh": false
}
```

### Sample Response: `POST /api/research`
```json
{
  "success": true,
  "id": "67cad1e2b4f9e31a89c1001a",
  "claim": "Smoking causes lung cancer.",
  "verdict": "SUPPORTS",
  "confidence": 0.901,
  "confidencePercent": 90,
  "probabilities": {
    "supports": 0.901,
    "refutes": 0.043,
    "notEnoughInfo": 0.056
  },
  "explanation": {
    "summary": "Extensive medical literature confirms that cigarette smoking is the primary cause of lung cancer.",
    "verdictExplanation": "Retrieved biomedical passages confirm a direct causal link between tobacco carcinogens and cellular lung mutations.",
    "keyEvidence": [
      "Cigarette smoking is responsible for approximately 85% of all lung cancer cases."
    ],
    "contradictoryEvidence": [],
    "uncertainty": "No significant evidentiary conflict detected among verified public health authorities."
  },
  "sourcesCount": 5,
  "evidenceCount": 8,
  "cached": false
}
```

---

## Installation & Setup

### Prerequisites
- **Node.js**: v18.0.0+ (Tested with v22.12.0)
- **npm**: v9.0.0+
- **Python**: 3.10+ (with virtual environment at `.venv`)
- **MongoDB**: Remote MongoDB Atlas Cluster (provided URI, no local database hosting required)
- **NVIDIA GPU** (Optional): CUDA 12.x supported (automatic CPU fallback if absent)

### 1. Python ML Service Setup
```powershell
cd ml/api
..\..\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd ../..
```

### 2. Node.js Backend Setup
```powershell
cd backend
npm install
cd ..
```

### 3. React Frontend Setup
```powershell
cd frontend
npm install
cd ..
```

---

## Environment Configuration

### Backend (`backend/.env`)
```env
PORT=5000
NODE_ENV=development
MONGO_URI=mongodb+srv://<username>:<password>@cluster0.b0eu6.mongodb.net/truthlens?retryWrites=true&w=majority
ML_SERVICE_URL=http://127.0.0.1:8000
CORS_ORIGIN=http://localhost:3000,http://127.0.0.1:3000

# Rate Limiting
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_RESEARCH=30
RATE_LIMIT_MAX_GENERAL=120
REQUEST_TIMEOUT_MS=60000
BODY_LIMIT=1mb

# Search Provider (duckduckgo, googlenews, wikipedia, newsapi, tavily, serper)
SEARCH_PROVIDER=duckduckgo
SEARCH_API_KEY=

# Retrieval Tuning
MAX_ARTICLES=8
MAX_PASSAGES=50
SEARCH_TIMEOUT_MS=12000
EXTRACTION_TIMEOUT_MS=8000
CACHE_TTL_HOURS=2

# LLM Explanation Layer (OpenAI or Gemini)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=
LLM_TIMEOUT_MS=15000
MAX_LLM_EVIDENCE=5
```

### Frontend (`frontend/.env`)
```env
VITE_API_URL=http://localhost:5000/api
```

---

## Running Locally

To run the complete platform locally, open 3 terminal windows:

### Terminal 1: Python FastAPI ML Service
```powershell
cd ml/api
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
*Health Check: `GET http://127.0.0.1:8000/health`*

### Terminal 2: Node.js Express Backend
```powershell
cd backend
npm run dev
```
*Health Check: `GET http://localhost:5000/api/health`*

### Terminal 3: React Frontend
```powershell
cd frontend
npm run dev
```
*Application UI: `http://localhost:3000`*

---

## Docker Deployment

The application services are containerized with multi-stage production Docker configurations:

```powershell
# Build and start services (FastAPI ML Service, Express Backend, React Frontend)
docker compose up --build -d

# Check running status
docker compose ps

# View unified container logs
docker compose logs -f
```

- **Frontend**: `http://localhost:3000`
- **Backend API**: `http://localhost:5000`
- **FastAPI ML Service**: `http://localhost:8000`

---

## Automated Testing & Benchmarks

### Backend Hardening & Security Test Suite
```powershell
cd backend
npm test
```
Executes 39 automated tests across health telemetry, boundary validation, SSRF ingress defenses, rate limiting, prompt injection resistance, and real-world claim categories.

### Frontend Production Build
```powershell
cd frontend
npm run build
```
Compiles Vite + Tailwind CSS v4 bundle with zero errors or unresolved dependencies.

### System Latency Waterfall
| Pipeline Stage | Measured Latency |
|---|---|
| Multi-Query Web Search | 0.8s – 1.4s |
| SSRF-Safe Article Extraction | 1.0s – 1.8s |
| Model A Relevance (50 Passages) | 0.4s – 0.8s |
| Model B Claim Verification | 0.5s – 1.0s |
| LLM Factual Synthesis | 0.8s – 1.5s (0.00s if unconfigured) |
| **Total Pipeline (Uncached)** | **2.5s – 4.0s** |
| **Cached Query Retrieval** | **< 15 ms** |

---

## Repository Structure

```
newproject/
├── backend/                       # Node.js + Express API Backend
│   ├── src/
│   │   ├── config/                # Environment configuration and database connection
│   │   ├── controllers/           # Route controllers (research, factCheck, health)
│   │   ├── middleware/            # Security (Helmet, CORS, rateLimiter, validateRequest, errorHandler)
│   │   ├── models/                # Mongoose FactCheck schema with compound indexes
│   │   ├── routes/                # Express API routes (/research, /fact-checks, /health)
│   │   └── services/              # Business logic (retrieval, articleExtractor, ML client, LLM synthesis)
│   ├── test_phase10_security.js   # Automated production verification test suite
│   ├── Dockerfile                 # Backend production container configuration
│   └── package.json
│
├── frontend/                      # React 18 + Vite SPA with Tailwind CSS v4
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/            # Navbar, Footer, ErrorBoundary, ErrorBanner, EmptyState
│   │   │   └── research/          # ClaimInput, VerdictCard, ProbabilityDistribution, Timeline
│   │   ├── pages/                 # ResearchPage, ReportPage, HistoryPage, ManualPage, ArchitecturePage
│   │   ├── services/              # API client and response normalization adapter
│   │   └── styles/                # Tailwind v4 theme, humanized tokens, Times New Roman typography
│   ├── index.html                 # HTML shell with Google Fonts Tinos & JetBrains Mono
│   ├── vite.config.js             # Vite 6 config with @tailwindcss/vite
│   ├── Dockerfile                 # Frontend production container configuration
│   └── package.json
│
├── ml/                            # Python ML & PyTorch Neural Inference Microservice
│   ├── api/
│   │   └── main.py                # FastAPI inference application with GPU health telemetry
│   ├── src/
│   │   ├── model_a.py             # RoBERTa evidence relevance model runner
│   │   ├── model_b.py             # SciFact claim verification model runner
│   │   └── fact_check_pipeline.py # Two-stage inference pipeline with CPU fallback
│   ├── models/                    # Fine-tuned model checkpoints (Model A & Model B)
│   ├── requirements.txt           # Python dependencies
│   └── Dockerfile                 # Python PyTorch CUDA container configuration
│
├── data/                          # Dataset splits and validation datasets (SciFact)
├── docker-compose.yml             # Container orchestration for MongoDB, ML, Backend, Frontend
├── LICENSE                        # Open-source MIT License
└── README.md                      # Master unified repository documentation
```

---

## License

This project is licensed under the [MIT License](LICENSE).
