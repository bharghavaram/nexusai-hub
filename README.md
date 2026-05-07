> **📅 Project Period:** May 2026 – Present &nbsp;|&nbsp; **Status:** Active &nbsp;|&nbsp; **Author:** [Bharghava Ram Vemuri](https://github.com/bharghavaram)

# 🤖 NexusAI Hub

> Production-grade Universal AI Intelligence Platform — LLM · NLP · SLM · GenAI · Automation in one system

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=FFD43B)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![OpenAI](https://img.shields.io/badge/GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?style=for-the-badge&logoColor=white)](https://spacy.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

---

## 🧩 Five Intelligence Modules

```
┌─────────────────────────────────────────────────────────────────┐
│                        NexusAI Hub                              │
├──────────┬──────────┬──────────┬───────────┬───────────────────┤
│   LLM    │   NLP    │   SLM    │  GenAI    │   Automation      │
│ GPT-4o   │  spaCy   │  BERT    │ Templates │ LLM Orchestrated  │
│ Claude   │ rule-NLP │  BART    │ Code Gen  │ Workflow Builder  │
│ RAG+FAISS│ Sentiment│ DistilBERT│ Reports  │ Step Execution    │
│ CoT      │ PII Det. │ Zero-Shot│ Creative  │ NL → Pipeline     │
│ Chat     │ Keywords │  NER     │ Translate │ Error Recovery    │
└──────────┴──────────┴──────────┴───────────┴───────────────────┘
                      FastAPI · REST · Pydantic · Docker
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/bharghavaram/nexusai-hub
cd nexusai-hub
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env        # Add your OPENAI_API_KEY
uvicorn main:app --reload
# Open http://localhost:8000
```

**Docker:**
```bash
docker build -t nexusai-hub .
docker run -p 8000:8000 --env-file .env nexusai-hub
```

---

## 📡 API Reference (40+ Endpoints)

### 🧠 LLM Module — `/api/v1/llm`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/llm/complete` | GPT-4o prompt completion |
| POST | `/llm/chat` | Multi-turn conversation with memory |
| POST | `/llm/reason` | Chain-of-Thought step-by-step reasoning |
| POST | `/llm/rag/ingest` | Upload documents to FAISS vector store |
| POST | `/llm/rag/query` | RAG-powered question answering |
| POST | `/llm/analyse` | Document analysis (summarise/extract/critique) |
| POST | `/llm/extract` | Structured JSON extraction from text |
| POST | `/llm/claude` | Anthropic Claude completion |

### 🔤 NLP Module — `/api/v1/nlp`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/nlp/entities` | Named entity recognition (spaCy) |
| POST | `/nlp/sentiment` | Sentiment analysis + sentence breakdown |
| POST | `/nlp/sentiment/batch` | Batch sentiment (up to 50 texts) |
| POST | `/nlp/summarise` | TF-IDF extractive summarisation |
| POST | `/nlp/keywords` | Keyword + bigram extraction |
| POST | `/nlp/pii` | PII detection + redaction |
| POST | `/nlp/language` | Language identification |
| POST | `/nlp/classify` | Text category classification |
| POST | `/nlp/readability` | Flesch reading ease score |
| POST | `/nlp/analyse/full` | All NLP tasks in one call |

### ⚡ SLM Module — `/api/v1/slm`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/slm/sentiment` | DistilBERT sentiment |
| POST | `/slm/zero-shot` | BART-MNLI zero-shot classification |
| POST | `/slm/ner` | BERT-NER named entity recognition |
| POST | `/slm/summarise` | BART abstractive summarisation |
| POST | `/slm/qa` | DistilBERT extractive Q&A |
| POST | `/slm/compare` | Jaccard text similarity |

### ✨ GenAI Module — `/api/v1/genai`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/genai/generate/blog` | SEO blog post generation |
| POST | `/genai/generate/email` | Professional email generation |
| POST | `/genai/generate/code` | Code generation (8 languages) |
| POST | `/genai/generate/social` | LinkedIn/Twitter/Instagram posts |
| POST | `/genai/generate/report` | Analysis/summary reports |
| POST | `/genai/generate/structured` | Structured JSON output |
| POST | `/genai/generate/creative` | Stories, poems, pitches |
| POST | `/genai/generate/template` | 8 built-in content templates |
| POST | `/genai/translate` | Multi-language translation |

### 🔄 Automation Module — `/api/v1/automation`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/automation/generate` | NL → workflow definition (GPT-4o) |
| POST | `/automation/execute` | Execute a workflow definition |
| POST | `/automation/run` | One-shot: describe → generate → execute |
| POST | `/automation/step/execute` | Test a single action step |
| GET | `/automation/actions` | List all 19 available actions |
| GET | `/automation/templates` | Pre-built workflow templates |

---

## 💡 Example Usage

### Chain-of-Thought Reasoning
```bash
curl -X POST "http://localhost:8000/api/v1/llm/reason" \
  -H "Content-Type: application/json" \
  -d '{"problem": "A train travels 120km in 1.5 hours. What is its average speed?"}'
```

### Full NLP Analysis
```bash
curl -X POST "http://localhost:8000/api/v1/nlp/analyse/full" \
  -H "Content-Type: application/json" \
  -d '{"text": "OpenAI released GPT-4o in 2024. It is an amazing multimodal model."}'
```

### Generate & Run a Workflow
```bash
curl -X POST "http://localhost:8000/api/v1/automation/run" \
  -H "Content-Type: application/json" \
  -d '{"description": "Analyse the sentiment of this text, extract keywords, and format as a report", "context_data": {"text": "AI is revolutionizing every industry in remarkable ways"}}'
```

### Generate Production Code
```bash
curl -X POST "http://localhost:8000/api/v1/genai/generate/code" \
  -H "Content-Type: application/json" \
  -d '{"description": "Redis-backed rate limiter with sliding window algorithm", "language": "Python", "style": "production"}'
```

---

## 🏗️ Architecture

```
nexusai-hub/
├── main.py                          # FastAPI app, all routers registered
├── app/
│   ├── core/config.py               # Pydantic Settings (env-based)
│   ├── services/
│   │   ├── llm_service.py           # GPT-4o, Claude, RAG, CoT, chat memory
│   │   ├── nlp_service.py           # spaCy NER, sentiment, PII, keywords
│   │   ├── slm_service.py           # HuggingFace Transformers local inference
│   │   ├── genai_service.py         # 9 content generation modes + templates
│   │   └── automation_service.py    # LLM workflow gen + 19-action executor
│   └── api/routes/
│       ├── llm.py · nlp.py · slm.py · genai.py · automation.py
├── tests/test_nexusai.py            # 40+ unit + integration tests
├── Dockerfile
└── .env.example
```

---

## 🔑 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | GPT-4o API key | For LLM + GenAI |
| `ANTHROPIC_API_KEY` | Claude API key | Optional |
| `HF_MODEL_SENTIMENT` | HuggingFace sentiment model | Optional |
| `SLM_DEVICE` | `cpu` or `cuda` | Optional |
| `RAG_TOP_K` | Documents to retrieve | Optional |

> **Note:** NLP, SLM (fallback mode), and Automation work **without any API keys**.

---

## 🧪 Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
# 40+ tests covering NLP, SLM, Automation, and all API endpoints
```
