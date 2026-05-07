"""NexusAI Hub — Full-Stack AI Platform: LLM + NLP + SLM + GenAI + Automation."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.api.routes.llm import router as llm_router
from app.api.routes.nlp import router as nlp_router
from app.api.routes.slm import router as slm_router
from app.api.routes.genai import router as genai_router
from app.api.routes.automation import router as automation_router
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NexusAI Hub",
    description="""
## 🤖 NexusAI Hub — Universal AI Intelligence Platform

A unified, production-grade AI platform combining five intelligence modules:

| Module | Technology | Capabilities |
|--------|-----------|-------------|
| **LLM** | GPT-4o + Claude | RAG Q&A, multi-turn chat, CoT reasoning, document analysis |
| **NLP** | spaCy + rule-based | NER, sentiment, keywords, PII detection, summarisation |
| **SLM** | HuggingFace Transformers | DistilBERT, BART, BERT-NER local inference |
| **GenAI** | GPT-4o + Claude | Blog, email, code, social, reports, creative writing |
| **Automation** | LLM orchestrated | Workflow generation + execution from plain English |
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

app.include_router(llm_router, prefix="/api/v1")
app.include_router(nlp_router, prefix="/api/v1")
app.include_router(slm_router, prefix="/api/v1")
app.include_router(genai_router, prefix="/api/v1")
app.include_router(automation_router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """<!DOCTYPE html>
<html><head>
<title>NexusAI Hub</title>
<style>
  body{font-family:Arial,sans-serif;background:linear-gradient(135deg,#0D1117 0%,#1a1a2e 50%,#16213e 100%);color:#e6edf3;margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;}
  .card{max-width:800px;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:16px;padding:48px;backdrop-filter:blur(10px);}
  h1{font-size:2.5rem;margin:0 0 8px;background:linear-gradient(90deg,#7B2FBE,#FF2E97,#00F5FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
  .subtitle{color:#8b949e;margin-bottom:32px;font-size:1.1rem;}
  .modules{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:32px;}
  .module{background:rgba(255,255,255,0.05);border-radius:12px;padding:20px;border-left:3px solid;}
  .llm{border-color:#7B2FBE;}.nlp{border-color:#FF2E97;}.slm{border-color:#00F5FF;}.genai{border-color:#FFB800;}.auto{border-color:#39FF14;}
  .module h3{margin:0 0 6px;font-size:1rem;}.module p{margin:0;font-size:0.85rem;color:#8b949e;}
  .links{display:flex;gap:16px;flex-wrap:wrap;}
  .btn{padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:600;font-size:0.9rem;transition:opacity 0.2s;}
  .btn-primary{background:linear-gradient(90deg,#7B2FBE,#FF2E97);color:#fff;}
  .btn-secondary{background:rgba(255,255,255,0.1);color:#e6edf3;border:1px solid rgba(255,255,255,0.2);}
  .btn:hover{opacity:0.85;}
  .badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:0.75rem;background:rgba(123,47,190,0.3);color:#B721FF;margin-bottom:16px;}
</style></head>
<body><div class="card">
  <span class="badge">v1.0.0 · Production Ready</span>
  <h1>NexusAI Hub</h1>
  <p class="subtitle">Universal AI Intelligence Platform — LLM · NLP · SLM · GenAI · Automation</p>
  <div class="modules">
    <div class="module llm"><h3>🧠 LLM Intelligence</h3><p>GPT-4o + Claude · RAG · CoT Reasoning · Multi-turn Chat · Document Analysis</p></div>
    <div class="module nlp"><h3>🔤 NLP Processing</h3><p>spaCy NER · Sentiment · Keywords · PII Detection · Language ID · Readability</p></div>
    <div class="module slm"><h3>⚡ Small LMs (SLM)</h3><p>DistilBERT · BART · BERT-NER · Zero-Shot · QA · Local Fast Inference</p></div>
    <div class="module genai"><h3>✨ GenAI Studio</h3><p>Blog · Email · Code · Social · Reports · Structured JSON · Creative Writing</p></div>
    <div class="module auto"><h3>🔄 Automation Engine</h3><p>NL → Workflow · LLM-generated pipelines · Step-by-step execution · Error recovery</p></div>
  </div>
  <div class="links">
    <a href="/docs" class="btn btn-primary">📖 Interactive API Docs</a>
    <a href="/redoc" class="btn btn-secondary">📚 ReDoc Reference</a>
    <a href="/api/v1/llm/health" class="btn btn-secondary">🔍 LLM Status</a>
    <a href="/api/v1/nlp/health" class="btn btn-secondary">🔍 NLP Status</a>
  </div>
</div></body></html>"""

@app.get("/health")
async def global_health():
    return {
        "status": "ok", "service": settings.APP_NAME, "version": settings.APP_VERSION,
        "modules": ["llm", "nlp", "slm", "genai", "automation"],
    }

logger.info(f"NexusAI Hub v{settings.APP_VERSION} started — 5 AI modules active")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
