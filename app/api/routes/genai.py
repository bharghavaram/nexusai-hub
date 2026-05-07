"""GenAI API Routes — Content generation, code, creative writing."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.services.genai_service import GenAIService

router = APIRouter(prefix="/genai", tags=["Generative AI"])
_svc = None

def get_service() -> GenAIService:
    global _svc
    if _svc is None:
        _svc = GenAIService()
    return _svc


class TemplateRequest(BaseModel):
    template: str = Field(..., description="Template name")
    variables: dict = Field(..., description="Template variable values")

class CodeRequest(BaseModel):
    description: str = Field(..., min_length=10)
    language: str = Field("Python", description="Target programming language")
    style: str = Field("production", description="production | beginner | minimal | test")

class BlogRequest(BaseModel):
    topic: str = Field(..., min_length=5)
    audience: str = Field("general")
    tone: str = Field("professional")
    length: str = Field("medium", description="short | medium | long")

class EmailRequest(BaseModel):
    purpose: str = Field(..., min_length=10)
    sender: str = Field("AI Assistant")
    recipient: str = Field("Recipient")
    tone: str = Field("professional")
    key_points: str = Field("")

class SocialRequest(BaseModel):
    topic: str = Field(..., min_length=5)
    platform: str = Field("linkedin", description="linkedin | twitter | instagram | facebook")
    goal: str = Field("engagement")

class ReportRequest(BaseModel):
    title: str = Field(..., min_length=3)
    data: str = Field(..., min_length=20, description="Data or context for the report")
    report_type: str = Field("analysis", description="analysis | summary | recommendation | comparison")
    audience: str = Field("executive")

class StructuredRequest(BaseModel):
    description: str = Field(..., min_length=5)
    output_schema: dict = Field(..., description="Target JSON schema")
    examples: Optional[list] = None

class CreativeRequest(BaseModel):
    prompt: str = Field(..., min_length=5)
    style: str = Field("short_story", description="short_story | poem | dialogue | pitch")
    genre: str = Field("general")

class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=5)
    target_language: str = Field(..., description="Target language name e.g. French, Spanish, Hindi")
    preserve_formatting: bool = Field(True)


@router.get("/health")
async def health():
    svc = get_service()
    return {"status": "ok", "openai_configured": bool(svc.openai),
            "anthropic_configured": bool(svc.anthropic),
            "available_templates": list(svc.available_templates.keys())}

@router.get("/templates")
async def list_templates():
    return {"templates": get_service().available_templates}

@router.post("/generate/template")
async def generate_from_template(req: TemplateRequest):
    try:
        return get_service().generate_from_template(req.template, req.variables)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))

@router.post("/generate/code")
async def generate_code(req: CodeRequest):
    valid_styles = {"production", "beginner", "minimal", "test"}
    if req.style not in valid_styles:
        raise HTTPException(400, f"style must be one of {valid_styles}")
    try:
        return get_service().generate_code(req.description, req.language, req.style)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

@router.post("/generate/blog")
async def generate_blog(req: BlogRequest):
    try:
        return get_service().generate_blog(req.topic, req.audience, req.tone, req.length)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

@router.post("/generate/email")
async def generate_email(req: EmailRequest):
    try:
        return get_service().generate_email(req.purpose, req.sender, req.recipient,
                                            req.tone, req.key_points)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

@router.post("/generate/social")
async def generate_social(req: SocialRequest):
    try:
        return get_service().generate_social(req.topic, req.platform, req.goal)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

@router.post("/generate/report")
async def generate_report(req: ReportRequest):
    valid_types = {"analysis", "summary", "recommendation", "comparison"}
    if req.report_type not in valid_types:
        raise HTTPException(400, f"report_type must be one of {valid_types}")
    try:
        return get_service().generate_report(req.title, req.data, req.report_type, req.audience)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

@router.post("/generate/structured")
async def generate_structured(req: StructuredRequest):
    try:
        return get_service().generate_structured(req.description, req.output_schema, req.examples)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

@router.post("/generate/creative")
async def generate_creative(req: CreativeRequest):
    valid_styles = {"short_story", "poem", "dialogue", "pitch"}
    if req.style not in valid_styles:
        raise HTTPException(400, f"style must be one of {valid_styles}")
    try:
        return get_service().generate_creative(req.prompt, req.style, req.genre)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

@router.post("/translate")
async def translate(req: TranslateRequest):
    try:
        return get_service().translate(req.text, req.target_language, req.preserve_formatting)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
