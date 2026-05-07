"""NLP Service — spaCy-powered NER, sentiment, summarisation, PII detection, keyword extraction."""
import re
import logging
import time
from collections import Counter
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class NLPService:
    def __init__(self):
        self._spacy_nlp = None
        self._spacy_loaded = False
        self._load_spacy()

    def _load_spacy(self):
        try:
            import spacy
            try:
                self._spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy en_core_web_sm not found — using blank model")
                self._spacy_nlp = spacy.blank("en")
            self._spacy_loaded = True
            logger.info("spaCy loaded")
        except ImportError:
            logger.warning("spaCy not installed — falling back to regex NLP")

    # ── Named Entity Recognition ─────────────────────────────────────────
    def extract_entities(self, text: str) -> dict:
        t0 = time.time()
        entities = []
        if self._spacy_loaded and self._spacy_nlp:
            doc = self._spacy_nlp(text[:10000])
            entities = [
                {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                for ent in doc.ents
            ]
            by_type = {}
            for e in entities:
                by_type.setdefault(e["label"], []).append(e["text"])
        else:
            # Regex fallback
            by_type = self._regex_ner(text)
            for label, items in by_type.items():
                for item in items:
                    entities.append({"text": item, "label": label})
        return {
            "entities": entities,
            "by_type": by_type if self._spacy_loaded else by_type,
            "entity_count": len(entities),
            "latency_ms": round((time.time() - t0) * 1000),
            "engine": "spaCy" if self._spacy_loaded else "regex"
        }

    def _regex_ner(self, text: str) -> dict:
        by_type: dict[str, list] = {}
        # Email
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails: by_type["EMAIL"] = list(set(emails))
        # Phone
        phones = re.findall(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
        if phones: by_type["PHONE"] = list(set(phones))
        # URL
        urls = re.findall(r'https?://[^\s]+', text)
        if urls: by_type["URL"] = list(set(urls))
        # Date
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', text)
        if dates: by_type["DATE"] = list(set(dates))
        # Money
        money = re.findall(r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:\.\d{2})?\s*(?:USD|EUR|GBP|INR)\b', text)
        if money: by_type["MONEY"] = list(set(money))
        return by_type

    # ── Sentiment Analysis ───────────────────────────────────────────────
    def analyse_sentiment(self, text: str) -> dict:
        t0 = time.time()
        # Rule-based lexicon sentiment (no model dependency)
        positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "outstanding", "brilliant", "superb", "perfect", "love", "best",
            "awesome", "happy", "joy", "success", "win", "positive", "nice",
            "helpful", "useful", "effective", "impressive", "innovative", "strong"
        }
        negative_words = {
            "bad", "terrible", "awful", "horrible", "poor", "worst", "hate",
            "fail", "failure", "error", "wrong", "broken", "slow", "useless",
            "annoying", "frustrating", "disappointing", "negative", "weak",
            "problem", "issue", "bug", "crash", "difficult", "hard", "ugly"
        }
        words = re.findall(r'\b\w+\b', text.lower())
        pos = sum(1 for w in words if w in positive_words)
        neg = sum(1 for w in words if w in negative_words)
        total = pos + neg if pos + neg > 0 else 1
        score = (pos - neg) / max(total, 1)
        # Normalise to 0-1
        compound = round((score + 1) / 2, 3)
        if compound >= 0.6:
            label, emoji = "POSITIVE", "😊"
        elif compound <= 0.4:
            label, emoji = "NEGATIVE", "😞"
        else:
            label, emoji = "NEUTRAL", "😐"
        # Sentence-level breakdown
        sentences = re.split(r'[.!?]+', text)
        sentence_scores = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 5:
                continue
            sw = re.findall(r'\b\w+\b', sent.lower())
            sp = sum(1 for w in sw if w in positive_words)
            sn = sum(1 for w in sw if w in negative_words)
            st = sp + sn if sp + sn > 0 else 1
            ss = round((sp - sn) / st, 2)
            sentence_scores.append({"text": sent[:100], "score": ss})
        return {
            "label": label, "emoji": emoji, "compound_score": compound,
            "positive_words": pos, "negative_words": neg,
            "sentences": sentence_scores[:5],
            "latency_ms": round((time.time() - t0) * 1000)
        }

    # ── Text Summarisation ───────────────────────────────────────────────
    def summarise(self, text: str, max_sentences: int = 5) -> dict:
        t0 = time.time()
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
        if not sentences:
            return {"summary": text[:500], "method": "truncation", "latency_ms": 0}
        # TF-IDF style sentence ranking
        words_all = re.findall(r'\b\w+\b', text.lower())
        stop_words = {"the","a","an","in","on","at","to","for","of","and","or","but","is","are","was","were","be","been","being","have","has","had","do","does","did","will","would","could","should","may","might","shall","it","its","this","that","these","those","i","we","you","he","she","they","what","which","who","how","when","where","why"}
        tf = Counter(w for w in words_all if w not in stop_words)
        total = max(tf.values()) if tf else 1
        tf_norm = {w: c / total for w, c in tf.items()}
        scored_sents = []
        for sent in sentences:
            words = re.findall(r'\b\w+\b', sent.lower())
            score = sum(tf_norm.get(w, 0) for w in words if w not in stop_words)
            scored_sents.append((score, sent))
        scored_sents.sort(key=lambda x: x[0], reverse=True)
        top = sorted(scored_sents[:max_sentences], key=lambda x: sentences.index(x[1]) if x[1] in sentences else 0)
        summary = " ".join(s for _, s in top)
        keywords = [w for w, _ in sorted(tf_norm.items(), key=lambda x: x[1], reverse=True)[:10]]
        return {
            "summary": summary, "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": round(len(summary) / max(len(text), 1), 2),
            "keywords": keywords, "method": "tf-idf-extractive",
            "latency_ms": round((time.time() - t0) * 1000)
        }

    # ── Keyword Extraction ───────────────────────────────────────────────
    def extract_keywords(self, text: str, top_n: int = 15) -> dict:
        t0 = time.time()
        stop_words = {"the","a","an","in","on","at","to","for","of","and","or","but","is","are","was","were","be","been","it","its","this","that","i","we","you","he","she","they","with","from","by","about","as","into","through","during","before","after","above","below","between","each","few","more","most","other","some","such","than","too","very","just","because","while","although","however","therefore","thus","hence","also","both","either","neither","nor","not","only","own","same","so","than","there","here","when","where","why","how","all","any","both","each","more","most","other","same","than","then","there","these","those","what","which","who","whom","whose","will","would","could","should"}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered = [w for w in words if w not in stop_words]
        freq = Counter(filtered)
        keywords = [{"keyword": w, "frequency": c, "score": round(c / len(filtered), 4)}
                    for w, c in freq.most_common(top_n)]
        # Bigrams
        bigrams = []
        for i in range(len(filtered) - 1):
            bg = f"{filtered[i]} {filtered[i+1]}"
            bigrams.append(bg)
        bigram_freq = Counter(bigrams)
        top_bigrams = [{"phrase": bg, "frequency": c}
                       for bg, c in bigram_freq.most_common(5) if c > 1]
        return {
            "keywords": keywords, "bigrams": top_bigrams,
            "total_words": len(words), "unique_words": len(set(filtered)),
            "latency_ms": round((time.time() - t0) * 1000)
        }

    # ── PII Detection ────────────────────────────────────────────────────
    def detect_pii(self, text: str) -> dict:
        t0 = time.time()
        pii_found = []
        patterns = {
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            "PHONE": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
            "CREDIT_CARD": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "IP_ADDRESS": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            "DATE_OF_BIRTH": r'\b(?:DOB|dob|date of birth|born)[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            "PASSPORT": r'\b[A-Z]{1,2}\d{6,9}\b',
            "AADHAAR": r'\b\d{4}\s?\d{4}\s?\d{4}\b',
        }
        for pii_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                pii_found.append({"type": pii_type, "value": match,
                                  "masked": self._mask(match)})
        redacted = text
        for item in pii_found:
            redacted = redacted.replace(item["value"], f"[{item['type']}_REDACTED]")
        return {
            "pii_detected": len(pii_found) > 0, "pii_count": len(pii_found),
            "items": pii_found, "redacted_text": redacted,
            "risk_level": "HIGH" if len(pii_found) > 3 else "MEDIUM" if pii_found else "LOW",
            "latency_ms": round((time.time() - t0) * 1000)
        }

    def _mask(self, value: str) -> str:
        if len(value) <= 4:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]

    # ── Language Detection ───────────────────────────────────────────────
    def detect_language(self, text: str) -> dict:
        t0 = time.time()
        # Heuristic language detection via character frequency
        lang_patterns = {
            "English": r'[a-zA-Z]',
            "Hindi/Devanagari": r'[\u0900-\u097F]',
            "Arabic": r'[\u0600-\u06FF]',
            "Chinese": r'[\u4e00-\u9fff]',
            "Japanese": r'[\u3040-\u309f\u30a0-\u30ff]',
            "Korean": r'[\uAC00-\uD7AF]',
            "Russian/Cyrillic": r'[\u0400-\u04FF]',
        }
        scores = {}
        for lang, pattern in lang_patterns.items():
            matches = len(re.findall(pattern, text))
            scores[lang] = matches / max(len(text), 1)
        detected = max(scores.items(), key=lambda x: x[1])
        return {
            "language": detected[0], "confidence": round(detected[1], 3),
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "latency_ms": round((time.time() - t0) * 1000)
        }

    # ── Text Classification ──────────────────────────────────────────────
    def classify_text(self, text: str, categories: list[str] = None) -> dict:
        t0 = time.time()
        if not categories:
            categories = ["Technology", "Business", "Health", "Science",
                          "Sports", "Politics", "Entertainment", "Education"]
        category_keywords = {
            "Technology": ["ai","ml","software","hardware","code","program","data","algorithm","neural","model","server","cloud","api","tech","digital","computer","robot","automation"],
            "Business": ["revenue","profit","market","company","startup","investment","growth","strategy","finance","budget","sales","customer","product","ceo","team"],
            "Health": ["medical","health","disease","treatment","doctor","hospital","patient","drug","therapy","clinical","symptom","diagnosis","medicine","wellness"],
            "Science": ["research","study","experiment","discovery","hypothesis","evidence","quantum","biology","physics","chemistry","lab","theory","scientific"],
            "Sports": ["game","team","player","score","match","league","tournament","win","goal","champion","athlete","sport","coach","season"],
            "Politics": ["government","policy","election","president","parliament","law","vote","political","party","senator","democracy","campaign"],
            "Entertainment": ["movie","music","film","actor","show","celebrity","concert","album","entertainment","artist","star","award","streaming"],
            "Education": ["school","university","student","teacher","course","learn","degree","academic","class","study","education","training","skill"],
        }
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_set = set(words)
        scores = {}
        for cat in categories:
            kws = category_keywords.get(cat, [cat.lower()])
            score = sum(1 for kw in kws if kw in word_set)
            scores[cat] = score
        total = sum(scores.values()) or 1
        probs = {cat: round(s / total, 3) for cat, s in scores.items()}
        best = max(probs.items(), key=lambda x: x[1])
        return {
            "label": best[0], "confidence": best[1],
            "probabilities": dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)),
            "latency_ms": round((time.time() - t0) * 1000)
        }

    # ── Readability Score ────────────────────────────────────────────────
    def readability(self, text: str) -> dict:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = re.findall(r'\b\w+\b', text)
        syllables = sum(self._count_syllables(w) for w in words)
        num_sentences = max(len(sentences), 1)
        num_words = max(len(words), 1)
        asl = num_words / num_sentences
        asw = syllables / num_words
        flesch = 206.835 - (1.015 * asl) - (84.6 * asw)
        flesch = max(0, min(100, flesch))
        if flesch >= 90: grade = "5th grade (Very Easy)"
        elif flesch >= 70: grade = "7th grade (Easy)"
        elif flesch >= 60: grade = "9th grade (Standard)"
        elif flesch >= 50: grade = "College (Fairly Difficult)"
        elif flesch >= 30: grade = "College Graduate (Difficult)"
        else: grade = "Professional (Very Difficult)"
        return {
            "flesch_reading_ease": round(flesch, 1), "grade_level": grade,
            "avg_sentence_length": round(asl, 1), "avg_syllables_per_word": round(asw, 2),
            "total_words": num_words, "total_sentences": num_sentences
        }

    def _count_syllables(self, word: str) -> int:
        word = word.lower()
        count = len(re.findall(r'[aeiou]+', word))
        if word.endswith('e') and len(word) > 2:
            count = max(1, count - 1)
        return max(1, count)
