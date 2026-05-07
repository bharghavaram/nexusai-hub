"""LLM Service — GPT-4o powered reasoning, RAG, multi-turn chat, CoT."""
import logging
import json
import time
from pathlib import Path
from typing import Optional
from openai import OpenAI
from anthropic import Anthropic
from app.core.config import settings

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Simple in-process conversation store."""
    def __init__(self):
        self._sessions: dict[str, list[dict]] = {}

    def get(self, session_id: str) -> list[dict]:
        return self._sessions.get(session_id, [])

    def add(self, session_id: str, role: str, content: str):
        self._sessions.setdefault(session_id, [])
        self._sessions[session_id].append({"role": role, "content": content})
        # Keep last 20 turns
        if len(self._sessions[session_id]) > 40:
            self._sessions[session_id] = self._sessions[session_id][-40:]

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)

    def all_sessions(self) -> list[str]:
        return list(self._sessions.keys())


class RAGEngine:
    """Lightweight FAISS-backed RAG engine."""
    def __init__(self):
        self._documents: list[dict] = []
        self._embeddings_client = None
        self._index = None
        self._faiss_available = False
        try:
            import faiss
            import numpy as np
            self._faiss = faiss
            self._np = np
            self._faiss_available = True
            Path(settings.FAISS_INDEX_PATH).mkdir(parents=True, exist_ok=True)
            logger.info("FAISS available — RAG enabled")
        except ImportError:
            logger.warning("FAISS not available — RAG will use keyword fallback")

    def _get_embedding(self, text: str) -> list[float]:
        if not self._embeddings_client:
            from openai import OpenAI
            self._embeddings_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        resp = self._embeddings_client.embeddings.create(
            model=settings.EMBED_MODEL, input=text[:8000]
        )
        return resp.data[0].embedding

    def ingest(self, texts: list[str], metadatas: list[dict] = None) -> int:
        metadatas = metadatas or [{} for _ in texts]
        if self._faiss_available and settings.OPENAI_API_KEY:
            embeddings = [self._get_embedding(t) for t in texts]
            arr = self._np.array(embeddings, dtype="float32")
            if self._index is None:
                self._index = self._faiss.IndexFlatL2(len(embeddings[0]))
            self._index.add(arr)
        for text, meta in zip(texts, metadatas):
            self._documents.append({"text": text, "meta": meta})
        logger.info(f"Ingested {len(texts)} documents (total: {len(self._documents)})")
        return len(texts)

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        top_k = top_k or settings.RAG_TOP_K
        if not self._documents:
            return []
        if self._faiss_available and self._index and settings.OPENAI_API_KEY:
            q_emb = self._np.array([self._get_embedding(query)], dtype="float32")
            k = min(top_k, len(self._documents))
            _, indices = self._index.search(q_emb, k)
            return [self._documents[i] for i in indices[0] if i < len(self._documents)]
        # Keyword fallback
        q_words = set(query.lower().split())
        scored = []
        for doc in self._documents:
            words = set(doc["text"].lower().split())
            score = len(q_words & words)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k] if _ > 0]

    @property
    def doc_count(self) -> int:
        return len(self._documents)


class LLMService:
    def __init__(self):
        self.openai = OpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
        self.anthropic = Anthropic(api_key=settings.ANTHROPIC_API_KEY) if settings.ANTHROPIC_API_KEY else None
        self.memory = ConversationMemory()
        self.rag = RAGEngine()

    # ── Core completion ──────────────────────────────────────────────────
    def complete(self, prompt: str, system: str = None, model: str = None,
                 temperature: float = None, max_tokens: int = None) -> dict:
        model = model or settings.LLM_MODEL
        temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        system = system or "You are NexusAI, a highly capable AI assistant."
        messages = [{"role": "user", "content": prompt}]
        t0 = time.time()
        if self.openai:
            resp = self.openai.chat.completions.create(
                model=model, messages=[{"role": "system", "content": system}] + messages,
                temperature=temperature, max_tokens=max_tokens
            )
            text = resp.choices[0].message.content
            usage = {"prompt_tokens": resp.usage.prompt_tokens,
                     "completion_tokens": resp.usage.completion_tokens,
                     "total_tokens": resp.usage.total_tokens}
        else:
            raise RuntimeError("No LLM API key configured.")
        return {"response": text, "model": model,
                "latency_ms": round((time.time() - t0) * 1000), "usage": usage}

    # ── Multi-turn chat ──────────────────────────────────────────────────
    def chat(self, session_id: str, message: str, system: str = None) -> dict:
        system = system or "You are NexusAI, a helpful conversational AI assistant."
        self.memory.add(session_id, "user", message)
        history = self.memory.get(session_id)
        t0 = time.time()
        if self.openai:
            resp = self.openai.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "system", "content": system}] + history,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            reply = resp.choices[0].message.content
            usage = {"total_tokens": resp.usage.total_tokens}
        else:
            raise RuntimeError("No LLM API key configured.")
        self.memory.add(session_id, "assistant", reply)
        return {"session_id": session_id, "reply": reply,
                "turns": len(history) // 2 + 1,
                "latency_ms": round((time.time() - t0) * 1000), "usage": usage}

    # ── Chain-of-Thought reasoning ───────────────────────────────────────
    def reason(self, problem: str) -> dict:
        cot_prompt = f"""Solve the following problem using step-by-step chain-of-thought reasoning.

PROBLEM: {problem}

Format your response as:
THINKING:
<step-by-step reasoning process>

ANSWER:
<final concise answer>

CONFIDENCE: <0-100>%
"""
        result = self.complete(cot_prompt, system="You are an expert analytical reasoner.", temperature=0.1)
        text = result["response"]
        thinking, answer, confidence = "", text, 80
        if "ANSWER:" in text:
            parts = text.split("ANSWER:")
            thinking = parts[0].replace("THINKING:", "").strip()
            rest = parts[1].strip()
            if "CONFIDENCE:" in rest:
                conf_parts = rest.split("CONFIDENCE:")
                answer = conf_parts[0].strip()
                try:
                    confidence = int(conf_parts[1].strip().replace("%", ""))
                except Exception:
                    pass
            else:
                answer = rest
        return {"problem": problem, "thinking": thinking, "answer": answer,
                "confidence": confidence, "latency_ms": result["latency_ms"]}

    # ── RAG Q&A ──────────────────────────────────────────────────────────
    def rag_query(self, question: str, top_k: int = None) -> dict:
        passages = self.rag.retrieve(question, top_k)
        context = "\n\n".join([f"[{i+1}] {p['text']}" for i, p in enumerate(passages)])
        if not context:
            return self.complete(question) | {"rag_used": False, "passages": 0}
        prompt = f"""Answer the question using the provided context. Cite source numbers [1],[2] etc.
If the context doesn't contain the answer, say so clearly.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        result = self.complete(prompt, system="You are a precise, factual Q&A assistant.", temperature=0.1)
        return result | {"rag_used": True, "passages": len(passages),
                         "sources": [p["meta"] for p in passages]}

    # ── Document analysis ────────────────────────────────────────────────
    def analyse_document(self, text: str, task: str = "summarise") -> dict:
        tasks = {
            "summarise": "Provide a comprehensive summary with key points, conclusions, and action items.",
            "extract_facts": "Extract all factual claims, statistics, dates, and named entities.",
            "qa_pairs": "Generate 10 insightful Q&A pairs from this document.",
            "critique": "Critically analyse this document: strengths, weaknesses, assumptions, gaps.",
            "translate_simple": "Rewrite this in plain English at an 8th-grade reading level.",
        }
        instruction = tasks.get(task, tasks["summarise"])
        prompt = f"""{instruction}

DOCUMENT:
{text[:12000]}

OUTPUT:"""
        return self.complete(prompt, temperature=0.2) | {"task": task, "doc_length": len(text)}

    # ── Structured extraction ────────────────────────────────────────────
    def extract_structured(self, text: str, schema: dict) -> dict:
        prompt = f"""Extract information from the text and return ONLY valid JSON matching this schema.

SCHEMA: {json.dumps(schema, indent=2)}

TEXT: {text[:6000]}

JSON OUTPUT:"""
        result = self.complete(prompt, temperature=0.0)
        raw = result["response"]
        try:
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            extracted = json.loads(raw.strip())
        except Exception:
            extracted = {"raw": result["response"], "parse_error": True}
        return {"extracted": extracted, "model": result["model"], "latency_ms": result["latency_ms"]}

    # ── Claude alternative ───────────────────────────────────────────────
    def complete_claude(self, prompt: str, system: str = None) -> dict:
        if not self.anthropic:
            raise RuntimeError("Anthropic API key not configured.")
        t0 = time.time()
        system = system or "You are NexusAI, a highly capable AI assistant."
        msg = self.anthropic.messages.create(
            model=settings.CLAUDE_MODEL, max_tokens=settings.LLM_MAX_TOKENS,
            system=system, messages=[{"role": "user", "content": prompt}]
        )
        return {"response": msg.content[0].text, "model": settings.CLAUDE_MODEL,
                "latency_ms": round((time.time() - t0) * 1000),
                "usage": {"input_tokens": msg.usage.input_tokens,
                          "output_tokens": msg.usage.output_tokens}}
