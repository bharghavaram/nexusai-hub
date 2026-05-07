"""SLM API Routes — Small Language Models via HuggingFace."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.services.slm_service import SLMService

router = APIRouter(prefix="/slm", tags=["Small Language Models"])
_svc = None

def get_service() -> SLMService:
    global _svc
    if _svc is None:
        _svc = SLMService()
    return _svc


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1)

class ZeroShotRequest(BaseModel):
    text: str = Field(..., min_length=5)
    labels: list[str] = Field(..., min_items=2, description="Candidate classification labels")

class QARequest(BaseModel):
    context: str = Field(..., min_length=20, description="Context passage")
    question: str = Field(..., min_length=5)

class SummariseRequest(BaseModel):
    text: str = Field(..., min_length=50)
    max_length: int = Field(130, ge=30, le=512)
    min_length: int = Field(30, ge=10, le=100)

class CompareRequest(BaseModel):
    text1: str = Field(..., min_length=5)
    text2: str = Field(..., min_length=5)

class BatchSentimentRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=20)


@router.get("/health")
async def health():
    svc = get_service()
    return {
        "status": "ok",
        "transformers_available": svc.available,
        "loaded_models": svc.loaded_models,
        "mode": "transformers" if svc.available else "rule-based-fallback"
    }

@router.post("/sentiment")
async def sentiment(req: TextRequest):
    return get_service().sentiment(req.text)

@router.post("/sentiment/batch")
async def batch_sentiment(req: BatchSentimentRequest):
    svc = get_service()
    results = []
    for text in req.texts:
        r = svc.sentiment(text)
        results.append({"text": text[:80], "label": r["label"], "score": r["score"]})
    return {"results": results, "total": len(results),
            "model": results[0].get("model", "unknown") if results else "none"}

@router.post("/zero-shot")
async def zero_shot(req: ZeroShotRequest):
    return get_service().zero_shot_classify(req.text, req.labels)

@router.post("/ner")
async def ner(req: TextRequest):
    return get_service().ner(req.text)

@router.post("/summarise")
async def summarise(req: SummariseRequest):
    return get_service().summarise(req.text, req.max_length, req.min_length)

@router.post("/qa")
async def question_answer(req: QARequest):
    return get_service().qa(req.context, req.question)

@router.post("/compare")
async def compare(req: CompareRequest):
    return get_service().compare_texts(req.text1, req.text2)

@router.get("/models")
async def list_models():
    svc = get_service()
    from app.core.config import settings
    return {
        "loaded": svc.loaded_models,
        "configured": {
            "sentiment": settings.HF_MODEL_SENTIMENT,
            "ner": settings.HF_MODEL_NER,
            "summarise": settings.HF_MODEL_SUMMARISE,
            "zero_shot": settings.HF_MODEL_ZERO_SHOT,
        },
        "device": settings.SLM_DEVICE
    }
