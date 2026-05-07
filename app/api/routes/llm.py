"""LLM API Routes."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.services.llm_service import LLMService

router = APIRouter(prefix="/llm", tags=["LLM Intelligence"])
_svc = None

def get_service() -> LLMService:
    global _svc
    if _svc is None:
        _svc = LLMService()
    return _svc


class CompleteRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The prompt to complete")
    system: Optional[str] = Field(None, description="System instruction")
    model: Optional[str] = Field(None, description="Override LLM model")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=8000)

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique conversation session ID")
    message: str = Field(..., min_length=1)
    system: Optional[str] = None

class ReasonRequest(BaseModel):
    problem: str = Field(..., min_length=10, description="Problem to reason about")

class RAGIngestRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, description="Documents to ingest")
    metadatas: Optional[list[dict]] = None

class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=5)
    top_k: Optional[int] = Field(None, ge=1, le=20)

class AnalyseDocRequest(BaseModel):
    text: str = Field(..., min_length=50)
    task: str = Field("summarise", description="summarise | extract_facts | qa_pairs | critique | translate_simple")

class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=10)
    schema: dict = Field(..., description="Target JSON schema to extract into")

class ClaudeRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    system: Optional[str] = None


@router.get("/health")
async def health():
    svc = get_service()
    return {"status": "ok", "openai_configured": bool(svc.openai),
            "anthropic_configured": bool(svc.anthropic),
            "rag_documents": svc.rag.doc_count,
            "active_sessions": len(svc.memory.all_sessions())}

@router.post("/complete")
async def complete(req: CompleteRequest):
    try:
        return get_service().complete(
            req.prompt, system=req.system, model=req.model,
            temperature=req.temperature, max_tokens=req.max_tokens
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/chat")
async def chat(req: ChatRequest):
    try:
        return get_service().chat(req.session_id, req.message, req.system)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

@router.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    get_service().memory.clear(session_id)
    return {"cleared": True, "session_id": session_id}

@router.get("/chat/sessions")
async def list_sessions():
    return {"sessions": get_service().memory.all_sessions()}

@router.post("/reason")
async def reason(req: ReasonRequest):
    try:
        return get_service().reason(req.problem)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

@router.post("/rag/ingest")
async def rag_ingest(req: RAGIngestRequest):
    try:
        count = get_service().rag.ingest(req.texts, req.metadatas)
        return {"ingested": count, "total_docs": get_service().rag.doc_count}
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/rag/query")
async def rag_query(req: RAGQueryRequest):
    try:
        return get_service().rag_query(req.question, req.top_k)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

@router.post("/analyse")
async def analyse_document(req: AnalyseDocRequest):
    valid_tasks = {"summarise", "extract_facts", "qa_pairs", "critique", "translate_simple"}
    if req.task not in valid_tasks:
        raise HTTPException(400, f"Invalid task. Choose from: {valid_tasks}")
    try:
        return get_service().analyse_document(req.text, req.task)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

@router.post("/extract")
async def extract_structured(req: ExtractRequest):
    try:
        return get_service().extract_structured(req.text, req.schema)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

@router.post("/claude")
async def complete_claude(req: ClaudeRequest):
    try:
        return get_service().complete_claude(req.prompt, req.system)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
