"""GenAI Service — Content generation, code generation, creative writing, structured output."""
import logging
import json
import re
import time
from openai import OpenAI
from anthropic import Anthropic
from app.core.config import settings

logger = logging.getLogger(__name__)

TEMPLATES = {
    "blog_post": """Write a professional, SEO-optimised blog post about: {topic}

Requirements:
- Engaging headline (H1)
- 3-4 main sections with H2 subheadings
- ~800 words
- Conversational yet authoritative tone
- Include a compelling introduction and conclusion with CTA
- Target audience: {audience}
- Tone: {tone}""",

    "email": """Write a professional email:
From: {sender}
To: {recipient}
Purpose: {purpose}
Tone: {tone}
Key points to include: {key_points}

Format: Subject line, greeting, body paragraphs, closing""",

    "product_description": """Write a compelling product description for:
Product: {product}
Target audience: {audience}
Key features: {features}
Price point: {price}

Include: headline, 2-3 benefit-focused paragraphs, bullet point features, CTA""",

    "linkedin_post": """Write a compelling LinkedIn post about: {topic}
Author background: {background}
Goal: {goal}
Tone: Professional yet personal
Length: 150-300 words
Include relevant hashtags (5-7)""",

    "press_release": """Write a professional press release:
Company: {company}
Announcement: {announcement}
Date: {date}
Quote from: {spokesperson}

Follow standard press release format: headline, dateline, intro paragraph, body, quote, boilerplate, contact info""",

    "job_description": """Write a comprehensive job description:
Role: {role}
Company: {company}
Department: {department}
Required skills: {skills}
Experience: {experience}

Include: overview, responsibilities (8-10 bullets), requirements, nice-to-haves, benefits""",

    "cover_letter": """Write a compelling cover letter:
Applicant: {name}
Role applying for: {role}
Company: {company}
Key experience: {experience}
Tone: Professional, enthusiastic

Keep to 3-4 paragraphs""",

    "code_review_comment": """Write professional code review comments for:
Language: {language}
Code snippet: {code}
Focus areas: correctness, performance, security, readability
Format as numbered, constructive comments with suggested fixes""",
}


class GenAIService:
    def __init__(self):
        self.openai = OpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
        self.anthropic = Anthropic(api_key=settings.ANTHROPIC_API_KEY) if settings.ANTHROPIC_API_KEY else None

    def _complete(self, prompt: str, system: str = None, temperature: float = None,
                  max_tokens: int = None, model: str = None) -> dict:
        model = model or settings.LLM_MODEL
        temperature = temperature if temperature is not None else settings.GENAI_TEMPERATURE
        max_tokens = max_tokens or settings.GENAI_MAX_TOKENS
        system = system or "You are a world-class creative and technical content generator."
        if not self.openai:
            raise RuntimeError("OPENAI_API_KEY not configured.")
        t0 = time.time()
        resp = self.openai.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens,
        )
        return {"content": resp.choices[0].message.content,
                "model": model, "latency_ms": round((time.time() - t0) * 1000),
                "usage": {"total_tokens": resp.usage.total_tokens}}

    # ── Template-based Content Generation ───────────────────────────────
    def generate_from_template(self, template_name: str, variables: dict) -> dict:
        if template_name not in TEMPLATES:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(TEMPLATES.keys())}")
        try:
            prompt = TEMPLATES[template_name].format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")
        result = self._complete(prompt)
        return result | {"template": template_name, "variables": variables}

    # ── Code Generation ──────────────────────────────────────────────────
    def generate_code(self, description: str, language: str = "Python",
                      style: str = "production") -> dict:
        style_instructions = {
            "production": "Write clean, production-ready code with type hints, docstrings, error handling, logging, and comprehensive inline comments.",
            "beginner": "Write simple, well-commented code with explanations of each step for a beginner.",
            "minimal": "Write minimal, concise code without extra comments.",
            "test": "Write comprehensive unit tests with pytest, covering happy path, edge cases, and error cases.",
        }
        instruction = style_instructions.get(style, style_instructions["production"])
        prompt = f"""{instruction}

Task: {description}
Language: {language}

Provide:
1. Complete working code
2. Brief explanation (2-3 sentences)
3. Example usage
4. Any dependencies required

Code:"""
        result = self._complete(prompt, system=f"You are an expert {language} software engineer.",
                                temperature=0.2)
        content = result["content"]
        # Extract code block
        code_match = re.search(r'```(?:\w+)?\n(.*?)```', content, re.DOTALL)
        code_only = code_match.group(1).strip() if code_match else content
        return result | {"language": language, "style": style, "code": code_only}

    # ── Blog / Article Generation ────────────────────────────────────────
    def generate_blog(self, topic: str, audience: str = "general",
                      tone: str = "professional", length: str = "medium") -> dict:
        lengths = {"short": "400-500 words", "medium": "700-900 words", "long": "1200-1500 words"}
        prompt = f"""Write a high-quality, SEO-optimised blog post.

Topic: {topic}
Target Audience: {audience}
Tone: {tone}
Length: {lengths.get(length, '700-900 words')}

Structure:
- Compelling H1 title
- Meta description (150 chars)
- Introduction with hook
- 3-4 H2 sections with content
- Conclusion with actionable takeaways
- 5 relevant SEO keywords used naturally

Blog Post:"""
        result = self._complete(prompt, temperature=0.75)
        return result | {"topic": topic, "audience": audience, "tone": tone, "length": length}

    # ── Email Generator ──────────────────────────────────────────────────
    def generate_email(self, purpose: str, sender: str = "AI Assistant",
                       recipient: str = "Recipient", tone: str = "professional",
                       key_points: str = "") -> dict:
        prompt = f"""Write a professional email:

Purpose: {purpose}
From: {sender}
To: {recipient}
Tone: {tone}
Key points: {key_points or 'As appropriate for the purpose'}

Provide:
- Subject: line
- Full email body
- Keep it concise and actionable"""
        result = self._complete(prompt, temperature=0.5)
        content = result["content"]
        subject = ""
        if "Subject:" in content:
            subject = content.split("Subject:")[1].split("\n")[0].strip()
        return result | {"purpose": purpose, "subject": subject}

    # ── Social Media Content ─────────────────────────────────────────────
    def generate_social(self, topic: str, platform: str = "linkedin",
                        goal: str = "engagement") -> dict:
        specs = {
            "linkedin": "Professional tone, 150-300 words, 5-7 hashtags, thought leadership angle",
            "twitter": "Punchy, 280 chars max, 2-3 hashtags, strong hook",
            "instagram": "Visual storytelling, emoji-friendly, 100-150 words, 10 hashtags",
            "facebook": "Conversational, 100-200 words, encourage discussion",
        }
        spec = specs.get(platform.lower(), specs["linkedin"])
        prompt = f"""Create a {platform} post about: {topic}
Goal: {goal}
Specs: {spec}

Provide the post content ready to copy-paste:"""
        result = self._complete(prompt, temperature=0.8)
        return result | {"topic": topic, "platform": platform, "goal": goal}

    # ── Report / Document Generation ─────────────────────────────────────
    def generate_report(self, title: str, data: str, report_type: str = "analysis",
                        audience: str = "executive") -> dict:
        types = {
            "analysis": "Analyse the data thoroughly, identify trends, anomalies, and insights",
            "summary": "Create a concise executive summary with key highlights",
            "recommendation": "Analyse the data and provide specific, actionable recommendations",
            "comparison": "Compare and contrast the elements, highlighting similarities and differences",
        }
        instruction = types.get(report_type, types["analysis"])
        prompt = f"""Generate a professional {report_type} report.

Title: {title}
Audience: {audience}
{instruction}

DATA/CONTEXT:
{data[:4000]}

Report format:
- Executive Summary
- Key Findings (bullet points)
- Detailed Analysis (2-3 sections)
- Conclusions
- Recommendations (if applicable)"""
        result = self._complete(prompt, temperature=0.3)
        return result | {"title": title, "report_type": report_type, "audience": audience}

    # ── Structured JSON Generation ────────────────────────────────────────
    def generate_structured(self, description: str, output_schema: dict,
                            examples: list = None) -> dict:
        example_str = ""
        if examples:
            example_str = f"\n\nExamples:\n{json.dumps(examples, indent=2)}"
        prompt = f"""Generate structured JSON data matching this schema exactly.

Description: {description}
Schema: {json.dumps(output_schema, indent=2)}{example_str}

Return ONLY valid JSON, no markdown, no explanation:"""
        result = self._complete(prompt, temperature=0.2)
        raw = result["content"].strip()
        if raw.startswith("```"):
            raw = re.sub(r'```\w*\n?', '', raw).replace('```', '').strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"raw": raw, "_parse_error": True}
        return result | {"structured": parsed, "schema": output_schema}

    # ── Creative Writing ─────────────────────────────────────────────────
    def generate_creative(self, prompt_text: str, style: str = "short_story",
                          genre: str = "general") -> dict:
        styles = {
            "short_story": "Write a compelling short story (300-500 words) with clear beginning, conflict, and resolution.",
            "poem": "Write an evocative poem with strong imagery, rhythm, and emotional resonance.",
            "dialogue": "Write a natural, character-revealing dialogue scene.",
            "pitch": "Write a compelling 60-second elevator pitch.",
        }
        instruction = styles.get(style, styles["short_story"])
        prompt = f"""{instruction}

Genre: {genre}
Prompt/Theme: {prompt_text}"""
        result = self._complete(prompt, temperature=0.9)
        return result | {"style": style, "genre": genre}

    # ── Translation ──────────────────────────────────────────────────────
    def translate(self, text: str, target_language: str,
                  preserve_formatting: bool = True) -> dict:
        fmt = "Preserve the original formatting, structure, and line breaks." if preserve_formatting else ""
        prompt = f"""Translate the following text to {target_language}. {fmt}
Provide ONLY the translation, no explanation.

TEXT:
{text}

TRANSLATION:"""
        result = self._complete(prompt, temperature=0.1)
        return result | {"source_length": len(text), "target_language": target_language}

    # ── Available Templates ──────────────────────────────────────────────
    @property
    def available_templates(self) -> dict:
        return {name: re.sub(r'\{(\w+)\}', r'<\1>', tmpl.strip()[:100] + "...")
                for name, tmpl in TEMPLATES.items()}
