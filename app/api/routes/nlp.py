"""NLP API Routes."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.services.nlp_service import NLPService

router = APIRouter(prefix="/nlp", tags=["NLP Processing"])
_svc = None

def get_service() -> NLPService:
    global _svc
    if _svc is None:
        _svc = NLPService()
    return _svc


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to process")

class SummariseRequest(BaseModel):
    text: str = Field(..., min_length=50)
    max_sentences: int = Field(5, ge=1, le=20)

class KeywordsRequest(BaseModel):
    text: str = Field(..., min_length=20)
    top_n: int = Field(15, ge=1, le=50)

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=10)
    categories: Optional[list[str]] = Field(None, description="Custom categories; defaults to 8 standard categories")

class BatchSentimentRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=50)

class FullAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=20, description="Run all NLP tasks on this text")


@router.get("/health")
async def health():
    svc = get_service()
    return {"status": "ok", "spacy_loaded": svc._spacy_loaded, "engine": "spaCy + rule-based"}

@router.post("/entities")
async def extract_entities(req: TextRequest):
    return get_service().extract_entities(req.text)

@router.post("/sentiment")
async def sentiment(req: TextRequest):
    return get_service().analyse_sentiment(req.text)

@router.post("/sentiment/batch")
async def batch_sentiment(req: BatchSentimentRequest):
    svc = get_service()
    results = []
    for text in req.texts:
        r = svc.analyse_sentiment(text)
        results.append({"text": text[:100], "label": r["label"],
                        "score": r["compound_score"], "emoji": r["emoji"]})
    labels = [r["label"] for r in results]
    return {"results": results, "total": len(results),
            "summary": {"POSITIVE": labels.count("POSITIVE"),
                        "NEGATIVE": labels.count("NEGATIVE"),
                        "NEUTRAL": labels.count("NEUTRAL")}}

@router.post("/summarise")
async def summarise(req: SummariseRequest):
    return get_service().summarise(req.text, req.max_sentences)

@router.post("/keywords")
async def keywords(req: KeywordsRequest):
    return get_service().extract_keywords(req.text, req.top_n)

@router.post("/pii")
async def detect_pii(req: TextRequest):
    return get_service().detect_pii(req.text)

@router.post("/language")
async def detect_language(req: TextRequest):
    return get_service().detect_language(req.text)

@router.post("/classify")
async def classify(req: ClassifyRequest):
    return get_service().classify_text(req.text, req.categories)

@router.post("/readability")
async def readability(req: TextRequest):
    return get_service().readability(req.text)

@router.post("/analyse/full")
async def full_analysis(req: FullAnalysisRequest):
    svc = get_service()
    text = req.text
    return {
        "text_length": len(text),
        "entities": svc.extract_entities(text),
        "sentiment": svc.analyse_sentiment(text),
        "keywords": svc.extract_keywords(text, top_n=10),
        "summary": svc.summarise(text),
        "pii": svc.detect_pii(text),
        "language": svc.detect_language(text),
        "classification": svc.classify_text(text),
        "readability": svc.readability(text),
    }
