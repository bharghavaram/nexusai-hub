"""Comprehensive tests for NexusAI Hub."""
import pytest
from fastapi.testclient import TestClient

# ── NLP Service Tests (no API key needed) ────────────────────────────────
from app.services.nlp_service import NLPService
from app.services.slm_service import SLMService
from app.services.automation_service import AutomationService, WorkflowStep, ACTIONS

@pytest.fixture
def nlp():
    return NLPService()

@pytest.fixture
def slm():
    return SLMService()

@pytest.fixture
def automation():
    return AutomationService()

@pytest.fixture
def client():
    from main import app
    return TestClient(app)

# ── NLP Tests ────────────────────────────────────────────────────────────
def test_nlp_sentiment_positive(nlp):
    result = nlp.analyse_sentiment("This is an amazing and wonderful product! I love it!")
    assert result["label"] == "POSITIVE"
    assert result["compound_score"] > 0.5

def test_nlp_sentiment_negative(nlp):
    result = nlp.analyse_sentiment("This is terrible and awful. Worst experience ever.")
    assert result["label"] == "NEGATIVE"
    assert result["compound_score"] < 0.5

def test_nlp_sentiment_fields(nlp):
    result = nlp.analyse_sentiment("The product is okay.")
    assert "label" in result
    assert "compound_score" in result
    assert "emoji" in result
    assert "latency_ms" in result

def test_nlp_extract_entities_email(nlp):
    result = nlp.extract_entities("Contact us at support@example.com or call +1-800-555-0100")
    assert result["entity_count"] >= 0
    assert "by_type" in result

def test_nlp_summarise(nlp):
    text = "Artificial intelligence is transforming every industry. From healthcare to finance, AI systems are helping humans make better decisions. Machine learning models can now process vast amounts of data in milliseconds. Natural language processing enables computers to understand human speech. These technologies are creating new opportunities while also raising important ethical questions."
    result = nlp.summarise(text)
    assert len(result["summary"]) > 0
    assert result["original_length"] == len(text)
    assert result["compression_ratio"] <= 1.0

def test_nlp_keywords(nlp):
    text = "Machine learning and artificial intelligence are revolutionizing data science. Deep learning models use neural networks to process large datasets and extract meaningful patterns."
    result = nlp.extract_keywords(text, top_n=5)
    assert len(result["keywords"]) <= 5
    assert all("keyword" in kw for kw in result["keywords"])

def test_nlp_pii_detection(nlp):
    text = "My email is john.doe@gmail.com and my SSN is 123-45-6789"
    result = nlp.detect_pii(text)
    assert result["pii_detected"] is True
    types = [item["type"] for item in result["items"]]
    assert "EMAIL" in types or "SSN" in types

def test_nlp_pii_clean_text(nlp):
    text = "The weather today is sunny and warm."
    result = nlp.detect_pii(text)
    assert result["pii_detected"] is False
    assert result["risk_level"] == "LOW"

def test_nlp_language_detection_english(nlp):
    result = nlp.detect_language("The quick brown fox jumps over the lazy dog")
    assert result["language"] == "English"

def test_nlp_classify_tech(nlp):
    text = "Machine learning algorithms and neural networks are transforming AI and software development"
    result = nlp.classify_text(text)
    assert result["label"] in ["Technology", "Science", "Education"]
    assert 0 <= result["confidence"] <= 1.0

def test_nlp_readability(nlp):
    text = "The cat sat on the mat. It was a nice day. The sun was bright."
    result = nlp.readability(text)
    assert "flesch_reading_ease" in result
    assert "grade_level" in result
    assert result["flesch_reading_ease"] >= 0

# ── SLM Tests ────────────────────────────────────────────────────────────
def test_slm_sentiment_fallback(slm):
    result = slm.sentiment("This product is amazing and I love it")
    assert result["label"] in {"POSITIVE", "NEGATIVE", "NEUTRAL"}
    assert 0 <= result["score"] <= 1.0

def test_slm_zero_shot_fallback(slm):
    result = slm.zero_shot_classify("Python is a great programming language", ["Technology", "Sports"])
    assert result["label"] in {"Technology", "Sports"}
    assert "all_scores" in result

def test_slm_compare_texts(slm):
    result = slm.compare_texts("I love machine learning", "I love deep learning")
    assert 0 <= result["jaccard_similarity"] <= 1.0
    assert "verdict" in result

def test_slm_compare_identical(slm):
    result = slm.compare_texts("hello world", "hello world")
    assert result["jaccard_similarity"] == 1.0

def test_slm_qa_fallback(slm):
    context = "Python was created by Guido van Rossum in 1991. It is known for its simple syntax."
    result = slm.qa(context, "Who created Python?")
    assert "answer" in result
    assert len(result["answer"]) > 0

# ── Automation Tests ─────────────────────────────────────────────────────
def test_action_text_transform_uppercase(automation):
    step = WorkflowStep(1, "text_transform", {"text": "hello world", "operation": "uppercase"})
    result = ACTIONS["text_transform"]({"text": "hello world", "operation": "uppercase"}, {})
    assert result["result"] == "HELLO WORLD"

def test_action_text_transform_word_count(automation):
    result = ACTIONS["text_transform"]({"text": "one two three four", "operation": "word_count"}, {})
    assert result["result"]["words"] == 4

def test_action_data_filter(automation):
    data = [{"age": 20}, {"age": 30}, {"age": 15}]
    result = ACTIONS["data_filter"]({"data": data, "key": "age", "operator": "greater_than", "value": 18}, {})
    assert result["result_count"] == 2

def test_action_data_filter_contains(automation):
    data = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Alice Smith"}]
    result = ACTIONS["data_filter"]({"data": data, "key": "name", "operator": "contains", "value": "Alice"}, {})
    assert result["result_count"] == 2

def test_action_data_transform_count(automation):
    result = ACTIONS["data_transform"]({"data": [1, 2, 3, 4, 5], "operation": "count"}, {})
    assert result["count"] == 5

def test_action_conditional_not_empty_true(automation):
    result = ACTIONS["conditional"]({"condition": "not_empty"}, {"last_output": [1, 2, 3]})
    assert result["result"] is True

def test_action_conditional_not_empty_false(automation):
    result = ACTIONS["conditional"]({"condition": "not_empty"}, {"last_output": []})
    assert result["result"] is False

def test_action_log_message(automation):
    result = ACTIONS["log_message"]({"message": "Test log", "level": "info"}, {})
    assert result["logged"] is True

def test_action_format_json(automation):
    result = ACTIONS["format_output"]({"format": "json", "data": {"key": "value"}}, {})
    assert "formatted" in result

def test_automation_list_templates(automation):
    templates = automation.list_templates()
    assert len(templates) >= 4
    assert "content_pipeline" in templates

def test_automation_available_actions(automation):
    actions = automation.available_actions
    assert "text_transform" in actions
    assert "data_filter" in actions
    assert "llm_complete" in actions

def test_automation_demo_workflow(automation):
    wf = automation._demo_workflow("Test automation workflow")
    assert "steps" in wf
    assert len(wf["steps"]) > 0

@pytest.mark.asyncio
async def test_automation_execute_simple():
    automation = AutomationService()
    wf_def = {
        "name": "Test Workflow",
        "steps": [
            {"step_id": 1, "action": "log_message",
             "params": {"message": "Starting test"}, "depends_on": []},
            {"step_id": 2, "action": "text_transform",
             "params": {"text": "hello", "operation": "uppercase"}, "depends_on": [1]},
        ]
    }
    record = await automation.execute_workflow(wf_def)
    assert record.status == "completed"
    assert len(record.steps) == 2

# ── API Tests ─────────────────────────────────────────────────────────────
def test_root_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200

def test_global_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "modules" in data

def test_nlp_health(client):
    resp = client.get("/api/v1/nlp/health")
    assert resp.status_code == 200

def test_slm_health(client):
    resp = client.get("/api/v1/slm/health")
    assert resp.status_code == 200

def test_automation_health(client):
    resp = client.get("/api/v1/automation/health")
    assert resp.status_code == 200

def test_api_nlp_sentiment(client):
    resp = client.post("/api/v1/nlp/sentiment", json={"text": "This is a great product!"})
    assert resp.status_code == 200
    assert resp.json()["label"] == "POSITIVE"

def test_api_nlp_keywords(client):
    resp = client.post("/api/v1/nlp/keywords", json={"text": "Machine learning and artificial intelligence are key technologies in modern software engineering and data science."})
    assert resp.status_code == 200
    assert "keywords" in resp.json()

def test_api_nlp_pii(client):
    resp = client.post("/api/v1/nlp/pii", json={"text": "Email me at test@example.com"})
    assert resp.status_code == 200
    assert "pii_detected" in resp.json()

def test_api_nlp_classify(client):
    resp = client.post("/api/v1/nlp/classify", json={"text": "Neural networks and deep learning are transforming AI research"})
    assert resp.status_code == 200
    assert "label" in resp.json()

def test_api_automation_actions(client):
    resp = client.get("/api/v1/automation/actions")
    assert resp.status_code == 200
    assert "actions" in resp.json()

def test_api_automation_templates(client):
    resp = client.get("/api/v1/automation/templates")
    assert resp.status_code == 200
    assert "templates" in resp.json()

def test_api_slm_compare(client):
    resp = client.post("/api/v1/slm/compare",
                       json={"text1": "AI is transforming tech", "text2": "AI is changing technology"})
    assert resp.status_code == 200
    assert "jaccard_similarity" in resp.json()

def test_api_genai_health(client):
    resp = client.get("/api/v1/genai/health")
    assert resp.status_code == 200

def test_api_genai_templates(client):
    resp = client.get("/api/v1/genai/templates")
    assert resp.status_code == 200
    assert "templates" in resp.json()

def test_api_llm_health(client):
    resp = client.get("/api/v1/llm/health")
    assert resp.status_code == 200
