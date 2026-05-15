# Contributing to NexusAI Hub

Thank you for your interest in contributing! This guide covers everything you need to get started.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Adding a New Endpoint](#adding-a-new-endpoint)
- [Adding a New Automation Action](#adding-a-new-automation-action)
- [Writing Tests](#writing-tests)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Reporting Bugs](#reporting-bugs)

---

## Code of Conduct

Be respectful, constructive, and inclusive. Focus on the work, not the person.

---

## How to Contribute

| Contribution Type | Welcome? |
|-------------------|----------|
| Bug fixes | ✅ Always |
| New API endpoints | ✅ With discussion |
| New automation actions | ✅ Always |
| New SLM models | ✅ With benchmarks |
| Documentation improvements | ✅ Always |
| Performance optimisations | ✅ With benchmarks |
| New LLM providers | ✅ With tests |

---

## Development Setup

```bash
# 1. Fork + clone
git clone https://github.com/YOUR_USERNAME/nexusai-hub.git
cd nexusai-hub

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov httpx

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Configure environment
cp .env.example .env
# Add OPENAI_API_KEY if testing LLM endpoints

# 6. Run tests to verify setup
pytest tests/ -v

# 7. Start the dev server
uvicorn main:app --reload
```

---

## Project Structure

```
app/services/      ← Business logic — add new AI capabilities here
app/api/routes/    ← FastAPI routers — add new HTTP endpoints here
tests/             ← pytest tests — every new feature needs a test
```

Key convention: **services contain all logic; routes only validate input and call services.**

---

## Adding a New Endpoint

1. Add the logic to the relevant `app/services/*.py` file
2. Add the route to the relevant `app/api/routes/*.py` file
3. Use Pydantic models for all request/response schemas
4. Add a test in `tests/test_nexusai.py`

**Example — adding `/nlp/emotion` endpoint:**

```python
# app/services/nlp_service.py — add method
def detect_emotion(self, text: str) -> dict:
    # your logic here
    return {"emotion": "joy", "score": 0.85}

# app/api/routes/nlp.py — add route
@router.post("/emotion")
async def detect_emotion(req: TextRequest):
    return get_service().detect_emotion(req.text)
```

---

## Adding a New Automation Action

Add a new handler in `app/services/automation_service.py`:

```python
# In the _execute_action method, add a new elif block:
elif action == "your_new_action":
    param1 = params.get("param1", "default")
    # your logic
    result = do_something(param1)
    return {"result": result, "action": action}
```

Then register it in `_get_available_actions()`:

```python
"your_new_action": {
    "description": "What this action does",
    "params": {"param1": "Description of param1"}
}
```

---

## Writing Tests

All tests go in `tests/test_nexusai.py`. Follow the existing pattern:

```python
class TestYourFeature:
    def test_basic_case(self):
        result = service.your_method("input text")
        assert "expected_key" in result
        assert result["expected_key"] is not None

    def test_edge_case_empty_input(self):
        # Test boundary conditions
        pass

    def test_api_endpoint(self, client):
        response = client.post("/api/v1/nlp/your-endpoint",
                               json={"text": "test input"})
        assert response.status_code == 200
        data = response.json()
        assert "expected_field" in data
```

**Run tests before submitting:**
```bash
pytest tests/ -v                          # All tests
pytest tests/ -v -k "your_feature"        # Specific tests
pytest tests/ --cov=app --cov-report=term # With coverage
```

---

## Pull Request Guidelines

1. **Branch name:** `feat/feature-name`, `fix/bug-description`, `docs/what-changed`
2. **Keep PRs focused** — one feature or fix per PR
3. **Tests required** — all new code must have tests
4. **No secrets** — never commit API keys or `.env` files
5. **Update README** if you add a new endpoint or environment variable

**PR description template:**
```
## What this PR does
Brief description of the change.

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation

## Testing
- [ ] Tests added
- [ ] All existing tests pass (`pytest tests/ -v`)

## Checklist
- [ ] Code follows the project structure conventions
- [ ] README updated if needed
```

---

## Reporting Bugs

Open a GitHub Issue with:

1. **Description** — what went wrong
2. **Steps to reproduce** — exact curl command or code that triggers it
3. **Expected vs actual** — what you expected vs what happened
4. **Environment** — Python version, OS, whether API keys are set

---

Thank you for helping make NexusAI Hub better! 🚀
