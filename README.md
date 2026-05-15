> **📅 Project Period:** May 2026 – Present &nbsp;|&nbsp; **Status:** Active &nbsp;|&nbsp; **Author:** [Bharghava Ram Vemuri](https://github.com/bharghavaram)

<div align="center">

# 🤖 NexusAI Hub

### Universal AI Intelligence Platform — LLM · NLP · SLM · GenAI · Automation

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=FFD43B)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![OpenAI](https://img.shields.io/badge/GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?style=for-the-badge&logoColor=white)](https://spacy.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-40%2B%20passing-brightgreen?style=for-the-badge)](tests/)

**40+ REST API Endpoints · 5 AI Modules · Production-Ready · Zero API Key for NLP/SLM**

[Problem Statement](#-problem-statement) · [Architecture](#-system-architecture) · [Quick Start](#-quick-start) · [API Docs](#-api-reference) · [Examples](#-sample-inputoutput-examples) · [Roadmap](#-roadmap)

</div>

---

## 🎯 Problem Statement

Modern AI applications require **multiple intelligence layers** working together — language understanding, content generation, NLP processing, local model inference, and workflow automation. But these capabilities are scattered across dozens of incompatible libraries, APIs, and services.

**The Problem:**
- Building a production AI app requires integrating OpenAI, HuggingFace, spaCy, workflow engines, and more separately
- Each service has different auth, rate limits, schemas, and failure modes
- No single platform unifies LLM reasoning + local NLP + SLM inference + GenAI content + automation workflows
- Developers spend weeks wiring these together instead of building product value

**NexusAI Hub solves this** by providing one unified FastAPI platform with 5 fully integrated AI intelligence modules, a single `.env` file for configuration, and 40+ production-ready endpoints accessible via REST — locally or via Docker.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          NexusAI Hub v1.0                                │
│                    Universal AI Intelligence Platform                     │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │     FastAPI Application        │
                    │   main.py · CORS · Pydantic    │
                    │   /docs (Swagger) · /redoc     │
                    └───────────────┬───────────────┘
                                    │
         ┌──────────┬───────────────┼───────────────┬──────────┐
         │          │               │               │          │
    ┌────▼────┐ ┌───▼────┐   ┌─────▼─────┐   ┌────▼────┐ ┌───▼──────────┐
    │   LLM   │ │  NLP   │   │    SLM    │   │  GenAI  │ │  Automation  │
    │ Module  │ │ Module │   │  Module   │   │  Module │ │   Module     │
    └────┬────┘ └───┬────┘   └─────┬─────┘   └────┬────┘ └───┬──────────┘
         │          │               │               │          │
    ┌────▼────┐ ┌───▼────┐   ┌─────▼─────┐   ┌────▼────┐ ┌───▼──────────┐
    │ GPT-4o  │ │ spaCy  │   │DistilBERT │   │ GPT-4o  │ │  GPT-4o      │
    │ Claude  │ │  NER   │   │   BART    │   │ Claude  │ │  Workflow    │
    │  FAISS  │ │ Sent.  │   │  BERT-NER │   │Template │ │  Generator   │
    │  RAG    │ │  PII   │   │  Zero-shot│   │  9 modes│ │  19 Actions  │
    │  CoT    │ │  Keys  │   │    Q&A    │   │ Trans.  │ │  Executor    │
    └─────────┘ └────────┘   └───────────┘   └─────────┘ └──────────────┘
         │                         │
    ┌────▼─────────────────────────▼─────────────┐
    │              External Services              │
    │  OpenAI API · Anthropic API · HuggingFace  │
    │         Hub (optional) · FAISS Index        │
    └─────────────────────────────────────────────┘
```

### Data Flow

```
User Request (JSON)
       │
       ▼
FastAPI Router → Pydantic Validation
       │
       ▼
Service Layer (llm/nlp/slm/genai/automation _service.py)
       │
       ├─► OpenAI API (GPT-4o)      ← if OPENAI_API_KEY set
       ├─► Anthropic API (Claude)   ← if ANTHROPIC_API_KEY set
       ├─► spaCy local model        ← always available
       ├─► HuggingFace Transformers ← local inference, no API key
       └─► Rule-based fallback      ← guaranteed response
       │
       ▼
Structured JSON Response → HTTP 200
```

---

## 📁 Project Structure

```
nexusai-hub/
│
├── main.py                          # FastAPI app entry point, router registration
│                                    # HTML dashboard at /, Swagger at /docs
│
├── app/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py                # Pydantic Settings — all env vars typed & validated
│   │
│   ├── services/                    # Business logic layer (no HTTP concerns)
│   │   ├── __init__.py
│   │   ├── llm_service.py           # GPT-4o, Claude, FAISS RAG, CoT, multi-turn chat
│   │   ├── nlp_service.py           # spaCy NER, sentiment, PII detection, keywords
│   │   ├── slm_service.py           # HuggingFace DistilBERT/BART/BERT-NER inference
│   │   ├── genai_service.py         # 9 content generation modes + 8 templates
│   │   └── automation_service.py    # LLM workflow generation + 19-action executor
│   │
│   └── api/
│       ├── __init__.py
│       └── routes/                  # FastAPI routers (one per module)
│           ├── __init__.py
│           ├── llm.py               # /api/v1/llm/* — 8 endpoints + health
│           ├── nlp.py               # /api/v1/nlp/* — 10 endpoints + health
│           ├── slm.py               # /api/v1/slm/* — 6 endpoints + health
│           ├── genai.py             # /api/v1/genai/* — 9 endpoints + health
│           └── automation.py        # /api/v1/automation/* — 6 endpoints + health
│
├── tests/
│   ├── __init__.py
│   └── test_nexusai.py              # 40+ unit + integration tests (pytest)
│
├── Dockerfile                       # Multi-stage production Docker build
├── .env.example                     # All env vars documented with defaults
├── requirements.txt                 # Pinned dependencies (26 packages)
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- An OpenAI API key (optional — NLP, SLM, Automation fallbacks work without it)

### Option 1 — Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/bharghavaram/nexusai-hub.git
cd nexusai-hub

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy language model
python -m spacy download en_core_web_sm

# 5. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (optional but enables LLM + GenAI)

# 6. Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Visit:
- **Interactive Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Dashboard:** http://localhost:8000

### Option 2 — Docker

```bash
# Build image
docker build -t nexusai-hub .

# Run with API key
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key_here \
  nexusai-hub

# Run with .env file
docker run -p 8000:8000 --env-file .env nexusai-hub
```

### Option 3 — Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  nexusai-hub:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/nlp/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
docker compose up -d
```

---

## 🤖 AI/ML Model Details

### LLM Module
| Component | Model | Provider | Purpose |
|-----------|-------|----------|---------|
| Chat Completion | `gpt-4o` | OpenAI API | Reasoning, generation, extraction |
| Claude Completion | `claude-3-5-sonnet-20241022` | Anthropic API | Alternative LLM |
| RAG Embeddings | `text-embedding-ada-002` | OpenAI API | Document vectorisation |
| Vector Store | FAISS (CPU) | Local | Similarity search over ingested docs |
| Chat Memory | In-memory session dict | Local | Multi-turn conversation state |

**RAG Pipeline:**
1. Documents ingested → chunked → embedded via `text-embedding-ada-002`
2. Stored in FAISS flat index (cosine similarity)
3. At query time: question embedded → top-K docs retrieved → injected into GPT-4o context

**Chain-of-Thought:**
- Structured system prompt instructs GPT-4o to output numbered reasoning steps
- Response parsed into `steps[]` + `final_answer` fields

### NLP Module
| Component | Algorithm | Dataset | Notes |
|-----------|-----------|---------|-------|
| NER | spaCy `en_core_web_sm` | OntoNotes 5.0 | 18 entity types |
| Sentiment | Lexicon-based (VADER-style) | Rule-based | No API key needed |
| Summarisation | TF-IDF extractive | N/A | Sentence scoring by term frequency |
| PII Detection | Regex + spaCy NER | N/A | Email, phone, SSN, CC, NER persons |
| Language Detection | Character n-gram heuristics | N/A | 10 languages |
| Readability | Flesch Reading Ease formula | N/A | Gunning Fog index |

### SLM Module (Local Inference — No API Key)
| Task | Model | Parameters | Hugging Face ID |
|------|-------|------------|-----------------|
| Sentiment | DistilBERT | 67M | `distilbert-base-uncased-finetuned-sst-2-english` |
| Zero-shot Classification | BART-MNLI | 407M | `facebook/bart-large-mnli` |
| NER | BERT-NER | 110M | `dbmdz/bert-large-cased-finetuned-conll03-english` |
| Summarisation | BART | 139M | `sshleifer/distilbart-cnn-12-6` |
| Question Answering | DistilBERT-QA | 67M | `distilbert-base-uncased-distilled-squad` |

All SLM models are downloaded on first use from the HuggingFace Hub and cached locally. Inference runs on CPU by default; set `SLM_DEVICE=cuda` for GPU acceleration.

### GenAI Module
- **Content Types:** Blog posts, professional emails, code (8 languages), social media (3 platforms), analysis reports, structured JSON, creative writing, template-based content, multi-language translation
- **Strategy:** Carefully engineered system prompts per content type; GPT-4o primary, Claude fallback
- **8 Built-in Templates:** Product launch, technical blog, weekly digest, cold outreach, incident report, API changelog, user interview, competitor analysis

### Automation Module
- **Workflow Generation:** GPT-4o converts natural language descriptions into structured JSON workflow definitions
- **19 Supported Actions:** `send_email`, `http_request`, `run_sql`, `log_message`, `set_variable`, `wait_seconds`, `format_text`, `send_notification`, `write_file`, `read_file`, `data_transform`, `filter_records`, `aggregate_data`, `send_webhook`, `cache_set`, `cache_get`, `counter_increment`, `validate_schema`, `nlp_analyse`
- **Execution Engine:** Step-by-step execution with context variable passing, error handling, and retry logic

---

## 📡 API Reference

All endpoints accept and return `application/json`. Base URL: `http://localhost:8000/api/v1`

### 🧠 LLM Module — `/api/v1/llm`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/llm/health` | Service health + RAG doc count + session count | No |
| POST | `/llm/complete` | GPT-4o prompt completion | Yes |
| POST | `/llm/chat` | Multi-turn conversation with session memory | Yes |
| DELETE | `/llm/chat/{session_id}` | Clear conversation session | No |
| POST | `/llm/reason` | Chain-of-Thought step-by-step reasoning | Yes |
| POST | `/llm/rag/ingest` | Upload documents to FAISS vector store | Yes |
| POST | `/llm/rag/query` | RAG-powered question answering | Yes |
| POST | `/llm/analyse` | Document analysis (summarise/extract/critique) | Yes |
| POST | `/llm/extract` | Structured JSON extraction from text | Yes |
| POST | `/llm/claude` | Anthropic Claude completion | Yes (Anthropic) |

### 🔤 NLP Module — `/api/v1/nlp`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/nlp/health` | Service health + spaCy model status | No |
| POST | `/nlp/entities` | Named entity recognition (18 types) | No |
| POST | `/nlp/sentiment` | Sentiment + per-sentence breakdown | No |
| POST | `/nlp/sentiment/batch` | Batch sentiment (up to 50 texts) | No |
| POST | `/nlp/summarise` | TF-IDF extractive summarisation | No |
| POST | `/nlp/keywords` | Keyword + bigram extraction with scores | No |
| POST | `/nlp/pii` | PII detection + automatic redaction | No |
| POST | `/nlp/language` | Language identification (10 languages) | No |
| POST | `/nlp/classify` | Text category classification | No |
| POST | `/nlp/readability` | Flesch reading ease + Gunning Fog | No |
| POST | `/nlp/analyse/full` | All NLP tasks in one call | No |

### ⚡ SLM Module — `/api/v1/slm`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/slm/health` | Loaded models + device info | No |
| POST | `/slm/sentiment` | DistilBERT sentiment classification | No |
| POST | `/slm/zero-shot` | BART-MNLI zero-shot classification | No |
| POST | `/slm/ner` | BERT NER entity extraction | No |
| POST | `/slm/summarise` | BART abstractive summarisation | No |
| POST | `/slm/qa` | DistilBERT extractive Q&A | No |
| POST | `/slm/compare` | Jaccard + token text similarity | No |

### ✨ GenAI Module — `/api/v1/genai`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/genai/health` | Service health + available templates | No |
| POST | `/genai/generate/blog` | SEO-optimised blog post | Yes |
| POST | `/genai/generate/email` | Professional email (6 tones) | Yes |
| POST | `/genai/generate/code` | Code generation (8 languages) | Yes |
| POST | `/genai/generate/social` | LinkedIn / Twitter / Instagram posts | Yes |
| POST | `/genai/generate/report` | Analysis / summary reports | Yes |
| POST | `/genai/generate/structured` | Structured JSON output | Yes |
| POST | `/genai/generate/creative` | Stories, poems, pitches | Yes |
| POST | `/genai/generate/template` | 8 built-in content templates | Yes |
| POST | `/genai/translate` | Multi-language translation | Yes |

### 🔄 Automation Module — `/api/v1/automation`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/automation/health` | Service health + action registry | No |
| POST | `/automation/generate` | NL description → workflow JSON (GPT-4o) | Yes |
| POST | `/automation/execute` | Execute a workflow definition | No |
| POST | `/automation/run` | One-shot: describe → generate → execute | Yes |
| POST | `/automation/step/execute` | Test a single workflow action step | No |
| GET | `/automation/actions` | List all 19 available actions + schemas | No |
| GET | `/automation/templates` | Pre-built workflow template library | No |

**Full interactive documentation available at:** `http://localhost:8000/docs`

---

## 💡 Sample Input/Output Examples

### 1. Chain-of-Thought Reasoning

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/llm/reason" \
  -H "Content-Type: application/json" \
  -d '{"problem": "A train travels 120km in 1.5 hours. What is its average speed?"}'
```

**Response:**
```json
{
  "problem": "A train travels 120km in 1.5 hours. What is its average speed?",
  "steps": [
    "Step 1: Identify the formula — Speed = Distance ÷ Time",
    "Step 2: Substitute values — Speed = 120 km ÷ 1.5 hours",
    "Step 3: Calculate — 120 ÷ 1.5 = 80",
    "Step 4: State units — 80 km/h"
  ],
  "final_answer": "The average speed of the train is 80 km/h.",
  "model": "gpt-4o",
  "tokens_used": 187
}
```

---

### 2. Full NLP Analysis

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/nlp/analyse/full" \
  -H "Content-Type: application/json" \
  -d '{"text": "Dr. Sarah Chen at OpenAI released GPT-4o in San Francisco on May 13, 2024. Her email is sarah@openai.com. The model achieves remarkable accuracy."}'
```

**Response:**
```json
{
  "entities": [
    {"text": "Dr. Sarah Chen", "label": "PERSON", "start": 0, "end": 14},
    {"text": "OpenAI", "label": "ORG", "start": 18, "end": 24},
    {"text": "GPT-4o", "label": "PRODUCT", "start": 33, "end": 39},
    {"text": "San Francisco", "label": "GPE", "start": 43, "end": 56},
    {"text": "May 13, 2024", "label": "DATE", "start": 60, "end": 72}
  ],
  "sentiment": {"label": "POSITIVE", "score": 0.82, "compound": 0.54},
  "keywords": ["GPT-4o", "OpenAI", "remarkable accuracy", "released"],
  "pii_detected": [
    {"type": "EMAIL", "value": "sarah@openai.com", "redacted": "[EMAIL]"}
  ],
  "language": "en",
  "readability": {"flesch_score": 48.2, "grade_level": "College"},
  "summary": "Dr. Sarah Chen at OpenAI released GPT-4o in San Francisco."
}
```

---

### 3. NL → Workflow Automation

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/automation/run" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Analyse the sentiment of the given text, extract keywords, then log a summary report",
    "context_data": {"text": "AI is transforming every industry at an unprecedented pace"}
  }'
```

**Response:**
```json
{
  "workflow": {
    "name": "Sentiment Analysis and Report Pipeline",
    "steps": [
      {"id": "step_1", "action": "nlp_analyse", "params": {"text": "{{text}}", "tasks": ["sentiment", "keywords"]}},
      {"id": "step_2", "action": "format_text", "params": {"template": "Sentiment: {{step_1.sentiment.label}} | Keywords: {{step_1.keywords}}"}},
      {"id": "step_3", "action": "log_message", "params": {"message": "{{step_2.result}}", "level": "info"}}
    ]
  },
  "execution": {
    "status": "completed",
    "steps_executed": 3,
    "steps_failed": 0,
    "results": {
      "step_1": {"sentiment": {"label": "POSITIVE", "score": 0.91}, "keywords": ["AI", "transforming", "unprecedented pace"]},
      "step_2": {"result": "Sentiment: POSITIVE | Keywords: AI, transforming, unprecedented pace"},
      "step_3": {"logged": true}
    }
  }
}
```

---

### 4. Generate Production Code

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/genai/generate/code" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Redis-backed rate limiter with sliding window algorithm",
    "language": "Python",
    "style": "production"
  }'
```

**Response:**
```json
{
  "language": "Python",
  "code": "import redis\nimport time\nfrom typing import Optional\n\nclass SlidingWindowRateLimiter:\n    def __init__(self, redis_client: redis.Redis, limit: int, window_seconds: int):\n        self.redis = redis_client\n        self.limit = limit\n        self.window = window_seconds\n\n    def is_allowed(self, key: str) -> tuple[bool, dict]:\n        now = time.time()\n        window_start = now - self.window\n        pipe = self.redis.pipeline()\n        pipe.zremrangebyscore(key, 0, window_start)\n        pipe.zadd(key, {str(now): now})\n        pipe.zcard(key)\n        pipe.expire(key, self.window)\n        _, _, count, _ = pipe.execute()\n        allowed = count <= self.limit\n        return allowed, {\"count\": count, \"limit\": self.limit, \"remaining\": max(0, self.limit - count)}\n",
  "explanation": "Uses Redis sorted sets with timestamps as scores. Removes expired entries, adds current request, counts remaining — all atomically via pipeline.",
  "model": "gpt-4o",
  "tokens_used": 312
}
```

---

### 5. Zero-Shot Classification (No API Key)

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/slm/zero-shot" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "My GPU memory keeps running out during model training",
    "labels": ["hardware issue", "software bug", "networking problem", "billing question"]
  }'
```

**Response:**
```json
{
  "text": "My GPU memory keeps running out during model training",
  "predictions": [
    {"label": "hardware issue", "score": 0.847},
    {"label": "software bug", "score": 0.112},
    {"label": "networking problem", "score": 0.026},
    {"label": "billing question", "score": 0.015}
  ],
  "best_label": "hardware issue",
  "confidence": 0.847,
  "model": "facebook/bart-large-mnli",
  "device": "cpu"
}
```

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` and configure:

```env
# ── LLM & GenAI (enables LLM, GenAI, Automation with AI planning) ──────────
OPENAI_API_KEY=sk-...           # Required for LLM + GenAI modules
ANTHROPIC_API_KEY=sk-ant-...    # Optional — enables Claude endpoints

# ── LLM Settings ────────────────────────────────────────────────────────────
DEFAULT_LLM_MODEL=gpt-4o        # gpt-4o | gpt-4-turbo | gpt-3.5-turbo
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=2000

# ── RAG Settings ─────────────────────────────────────────────────────────────
RAG_TOP_K=5                     # Documents retrieved per query
EMBEDDING_MODEL=text-embedding-ada-002

# ── SLM / HuggingFace Settings ───────────────────────────────────────────────
SLM_DEVICE=cpu                  # cpu | cuda | mps (Apple Silicon)
HF_MODEL_SENTIMENT=distilbert-base-uncased-finetuned-sst-2-english
HF_MODEL_ZERO_SHOT=facebook/bart-large-mnli
HF_MODEL_NER=dbmdz/bert-large-cased-finetuned-conll03-english
HF_MODEL_SUMMARISE=sshleifer/distilbart-cnn-12-6
HF_MODEL_QA=distilbert-base-uncased-distilled-squad

# ── Server Settings ──────────────────────────────────────────────────────────
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
```

> **Works without any API key:** NLP (spaCy), SLM (HuggingFace local), Automation (rule-based fallback)

---

## 🚢 Deployment Instructions

### Deploy to AWS EC2

```bash
# On EC2 instance (Ubuntu 22.04 LTS, t3.medium minimum)
sudo apt update && sudo apt install -y python3.11 python3.11-venv git docker.io
sudo systemctl start docker

git clone https://github.com/bharghavaram/nexusai-hub.git
cd nexusai-hub
cp .env.example .env && nano .env   # Add your API keys

docker build -t nexusai-hub .
docker run -d -p 80:8000 --env-file .env --name nexusai nexusai-hub
```

### Deploy to Railway / Render / Fly.io

1. Fork this repo
2. Connect your Railway/Render account to the repo
3. Add environment variables in the dashboard
4. Set **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Deploy — both platforms auto-detect the `requirements.txt`

### Deploy to Google Cloud Run

```bash
# Build & push to Artifact Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT/nexusai-hub

# Deploy
gcloud run deploy nexusai-hub \
  --image gcr.io/YOUR_PROJECT/nexusai-hub \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your_key
```

### Deploy to Azure Container Apps

```bash
az containerapp create \
  --name nexusai-hub \
  --resource-group myRG \
  --image nexusai-hub:latest \
  --target-port 8000 \
  --ingress external \
  --env-vars OPENAI_API_KEY=your_key
```

### Kubernetes (Helm / kubectl)

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexusai-hub
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nexusai-hub
  template:
    spec:
      containers:
      - name: nexusai-hub
        image: nexusai-hub:latest
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: nexusai-secrets
        resources:
          requests: { memory: "512Mi", cpu: "250m" }
          limits:   { memory: "2Gi",  cpu: "1000m" }
        livenessProbe:
          httpGet: { path: /api/v1/nlp/health, port: 8000 }
          initialDelaySeconds: 30
```

---

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/ -v -k "nlp"
pytest tests/ -v -k "automation"

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=app --cov-report=html
```

**Test coverage:**
- ✅ NLP: entity extraction, sentiment, PII, keywords, readability, language detection
- ✅ SLM: sentiment, zero-shot, NER, QA, text comparison (with mock transformers)
- ✅ Automation: workflow generation, step execution, all 19 action handlers
- ✅ GenAI: content generation for all 9 modes (with mock OpenAI)
- ✅ API: all routes return correct status codes and response schemas

---

## 📦 Requirements

All dependencies pinned for reproducibility:

```
fastapi==0.115.5           # Web framework
uvicorn[standard]==0.32.1  # ASGI server
pydantic==2.10.3           # Data validation
pydantic-settings==2.7.0   # Settings management
openai==1.59.3             # GPT-4o API
anthropic==0.42.0          # Claude API
langchain==0.3.12          # LLM orchestration
langchain-openai==0.3.0
langchain-community==0.3.12
faiss-cpu==1.9.0.post1     # Vector similarity search
spacy==3.8.3               # NLP pipeline
transformers==4.47.1       # HuggingFace models
torch==2.5.1               # PyTorch (SLM inference)
sentence-transformers==3.3.1
huggingface-hub==0.27.0
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.6.0
httpx==0.28.1              # Async HTTP client
python-multipart==0.0.20
python-dotenv==1.0.1
sqlalchemy==2.0.36
aiosqlite==0.20.0
rich==13.9.4
pytest==8.3.4
pytest-asyncio==0.25.0
```

---

## 🗺️ Roadmap

### v1.1 — Q3 2026
- [ ] **Streaming responses** — Server-Sent Events for real-time LLM output
- [ ] **WebSocket chat** — Persistent bi-directional chat sessions
- [ ] **Rate limiting** — Per-API-key request throttling with Redis
- [ ] **Auth middleware** — JWT-based API key management

### v1.2 — Q4 2026
- [ ] **Multi-modal support** — Image analysis via GPT-4 Vision + OCR
- [ ] **Voice input** — Whisper speech-to-text integration
- [ ] **Persistent vector store** — PostgreSQL + pgvector replacing in-memory FAISS
- [ ] **Workflow history** — SQLite-backed automation execution log

### v2.0 — 2027
- [ ] **Agent loop** — Fully autonomous multi-step task execution with tool use
- [ ] **Plugin system** — Community-contributed automation actions
- [ ] **Fine-tuning API** — PEFT/LoRA fine-tuning endpoint via HuggingFace PEFT
- [ ] **React dashboard** — Full web UI for all 5 modules
- [ ] **Kubernetes Helm chart** — Production-grade K8s deployment package

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

**Quick contribution steps:**
1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes with tests
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Bharghava Ram Vemuri**

[![GitHub](https://img.shields.io/badge/GitHub-bharghavaram-181717?style=flat&logo=github)](https://github.com/bharghavaram)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-bharghavaram--vemuri-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/bharghavaram-vemuri)

---

<div align="center">

**⭐ Star this repo if you find it useful — it helps others discover it!**

Made with ❤️ by [Bharghava Ram Vemuri](https://github.com/bharghavaram)

</div>
