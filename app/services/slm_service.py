"""SLM Service — Small Language Models via HuggingFace Transformers for fast local inference."""
import logging
import time
import re
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class SLMService:
    """
    Small Language Model service using HuggingFace Transformers.
    Loads models lazily on first use to minimise startup time.
    Falls back to rule-based methods if transformers not available.
    """
    def __init__(self):
        self._pipeline = None
        self._transformers_available = False
        self._loaded_models: dict = {}
        try:
            from transformers import pipeline
            self._hf_pipeline = pipeline
            self._transformers_available = True
            logger.info("HuggingFace Transformers available — SLM enabled")
        except ImportError:
            logger.warning("transformers not installed — SLM using lightweight fallback")

    def _get_pipeline(self, task: str, model: str):
        key = f"{task}::{model}"
        if key not in self._loaded_models:
            if not self._transformers_available:
                return None
            logger.info(f"Loading SLM model: {model} for task: {task}")
            try:
                pipe = self._hf_pipeline(task, model=model, device=-1)
                self._loaded_models[key] = pipe
            except Exception as e:
                logger.error(f"Failed to load {model}: {e}")
                return None
        return self._loaded_models.get(key)

    # ── Sentiment (DistilBERT) ───────────────────────────────────────────
    def sentiment(self, text: str) -> dict:
        t0 = time.time()
        pipe = self._get_pipeline("sentiment-analysis", settings.HF_MODEL_SENTIMENT)
        if pipe:
            result = pipe(text[:512])[0]
            label = result["label"]
            score = round(result["score"], 4)
        else:
            # Fallback: simple lexicon
            pos_words = {"good","great","excellent","amazing","love","perfect","best","happy","wonderful"}
            neg_words = {"bad","terrible","awful","hate","worst","horrible","poor","fail","broken"}
            words = set(text.lower().split())
            pos = len(words & pos_words)
            neg = len(words & neg_words)
            if pos > neg:
                label, score = "POSITIVE", round(0.6 + 0.1 * min(pos, 4), 2)
            elif neg > pos:
                label, score = "NEGATIVE", round(0.6 + 0.1 * min(neg, 4), 2)
            else:
                label, score = "NEUTRAL", 0.52
        return {"label": label, "score": score,
                "model": settings.HF_MODEL_SENTIMENT if self._transformers_available else "lexicon-fallback",
                "latency_ms": round((time.time() - t0) * 1000)}

    # ── Zero-Shot Classification (BART-MNLI) ────────────────────────────
    def zero_shot_classify(self, text: str, labels: list[str]) -> dict:
        t0 = time.time()
        if not labels:
            return {"error": "No labels provided"}
        pipe = self._get_pipeline("zero-shot-classification", settings.HF_MODEL_ZERO_SHOT)
        if pipe:
            result = pipe(text[:512], candidate_labels=labels)
            return {
                "label": result["labels"][0],
                "score": round(result["scores"][0], 4),
                "all_scores": {l: round(s, 4) for l, s in zip(result["labels"], result["scores"])},
                "model": settings.HF_MODEL_ZERO_SHOT,
                "latency_ms": round((time.time() - t0) * 1000)
            }
        # Keyword fallback
        text_lower = text.lower()
        scores = {}
        for label in labels:
            words = re.findall(r'\b\w+\b', label.lower())
            score = sum(1 for w in words if w in text_lower) / max(len(words), 1)
            scores[label] = round(score, 3)
        total = sum(scores.values()) or 1
        norm = {k: round(v / total, 4) for k, v in scores.items()}
        best = max(norm, key=norm.get)
        return {"label": best, "score": norm[best], "all_scores": norm,
                "model": "keyword-fallback",
                "latency_ms": round((time.time() - t0) * 1000)}

    # ── NER (BERT-NER) ──────────────────────────────────────────────────
    def ner(self, text: str) -> dict:
        t0 = time.time()
        pipe = self._get_pipeline("ner", settings.HF_MODEL_NER)
        if pipe:
            raw = pipe(text[:512])
            entities = []
            current = None
            for token in raw:
                tag = token.get("entity", "O")
                word = token.get("word", "")
                if tag.startswith("B-"):
                    if current:
                        entities.append(current)
                    current = {"text": word.replace("##", ""), "label": tag[2:],
                               "score": round(token["score"], 3)}
                elif tag.startswith("I-") and current:
                    current["text"] += word.replace("##", "")
                    current["score"] = round((current["score"] + token["score"]) / 2, 3)
                else:
                    if current:
                        entities.append(current)
                        current = None
            if current:
                entities.append(current)
        else:
            # Regex fallback
            entities = []
            for email in re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text):
                entities.append({"text": email, "label": "EMAIL", "score": 1.0})
            for phone in re.findall(r'\b\+?[\d\s\-().]{10,}\b', text):
                entities.append({"text": phone.strip(), "label": "PHONE", "score": 0.9})
        by_label: dict[str, list] = {}
        for e in entities:
            by_label.setdefault(e["label"], []).append(e["text"])
        return {"entities": entities, "by_label": by_label, "count": len(entities),
                "model": settings.HF_MODEL_NER if self._transformers_available else "regex-fallback",
                "latency_ms": round((time.time() - t0) * 1000)}

    # ── Text Summarisation (BART) ────────────────────────────────────────
    def summarise(self, text: str, max_length: int = 130, min_length: int = 30) -> dict:
        t0 = time.time()
        pipe = self._get_pipeline("summarization", settings.HF_MODEL_SUMMARISE)
        if pipe and len(text) > 100:
            try:
                result = pipe(text[:1024], max_length=max_length,
                              min_length=min_length, do_sample=False)[0]
                summary = result["summary_text"]
                return {"summary": summary, "original_length": len(text),
                        "summary_length": len(summary),
                        "model": settings.HF_MODEL_SUMMARISE,
                        "latency_ms": round((time.time() - t0) * 1000)}
            except Exception as e:
                logger.error(f"BART summarisation failed: {e}")
        # Extractive fallback: first + last sentence + middle key sentence
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 30]
        if len(sentences) <= 3:
            summary = " ".join(sentences)
        else:
            mid = sentences[len(sentences) // 2]
            summary = f"{sentences[0]}. {mid}. {sentences[-1]}."
        return {"summary": summary, "original_length": len(text),
                "summary_length": len(summary), "model": "extractive-fallback",
                "latency_ms": round((time.time() - t0) * 1000)}

    # ── Question Answering ───────────────────────────────────────────────
    def qa(self, context: str, question: str) -> dict:
        t0 = time.time()
        pipe = self._get_pipeline(
            "question-answering",
            "distilbert-base-cased-distilled-squad"
        )
        if pipe:
            result = pipe(question=question, context=context[:512])
            return {"answer": result["answer"], "score": round(result["score"], 4),
                    "start": result["start"], "end": result["end"],
                    "model": "distilbert-squad",
                    "latency_ms": round((time.time() - t0) * 1000)}
        # Keyword search fallback
        q_words = set(re.findall(r'\b\w+\b', question.lower()))
        sentences = re.split(r'[.!?]+', context)
        best, best_score = context[:200], 0
        for sent in sentences:
            s_words = set(re.findall(r'\b\w+\b', sent.lower()))
            score = len(q_words & s_words)
            if score > best_score:
                best_score, best = score, sent.strip()
        return {"answer": best, "score": 0.0, "model": "keyword-fallback",
                "latency_ms": round((time.time() - t0) * 1000)}

    # ── Text-to-Text Comparison ──────────────────────────────────────────
    def compare_texts(self, text1: str, text2: str) -> dict:
        t0 = time.time()
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        intersection = words1 & words2
        union = words1 | words2
        jaccard = len(intersection) / max(len(union), 1)
        only_in_1 = words1 - words2
        only_in_2 = words2 - words1
        return {
            "jaccard_similarity": round(jaccard, 4),
            "similarity_percent": round(jaccard * 100, 1),
            "shared_words": len(intersection),
            "unique_to_text1": len(only_in_1),
            "unique_to_text2": len(only_in_2),
            "verdict": "Very Similar" if jaccard > 0.7 else "Somewhat Similar" if jaccard > 0.4 else "Different",
            "latency_ms": round((time.time() - t0) * 1000)
        }

    @property
    def loaded_models(self) -> list[str]:
        return list(self._loaded_models.keys())

    @property
    def available(self) -> bool:
        return self._transformers_available
