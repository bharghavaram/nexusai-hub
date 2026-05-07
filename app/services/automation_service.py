"""Automation Service — AI-powered workflow builder, pipeline execution, task orchestration."""
import logging
import json
import time
import uuid
import asyncio
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from openai import OpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStep:
    def __init__(self, step_id: int, action: str, params: dict,
                 description: str = "", depends_on: list[int] = None):
        self.step_id = step_id
        self.action = action
        self.params = params
        self.description = description
        self.depends_on = depends_on or []
        self.status = TaskStatus.PENDING
        self.result: Any = None
        self.error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return round((self.completed_at - self.started_at) * 1000)
        return None

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id, "action": self.action,
            "description": self.description, "params": self.params,
            "status": self.status, "result": self.result,
            "error": self.error, "duration_ms": self.duration_ms,
        }


class WorkflowRecord:
    def __init__(self, workflow_id: str, name: str, steps: list[WorkflowStep]):
        self.workflow_id = workflow_id
        self.name = name
        self.steps = steps
        self.status = TaskStatus.PENDING
        self.created_at = datetime.utcnow().isoformat()
        self.completed_at: Optional[str] = None
        self.total_duration_ms: Optional[float] = None
        self.metadata: dict = {}

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id, "name": self.name,
            "status": self.status,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "total_duration_ms": self.total_duration_ms,
            "step_count": len(self.steps),
            "completed_steps": sum(1 for s in self.steps if s.status == TaskStatus.COMPLETED),
            "failed_steps": sum(1 for s in self.steps if s.status == TaskStatus.FAILED),
        }


# ── Built-in Action Handlers ──────────────────────────────────────────────
class ActionRegistry:
    """Registry of built-in automation actions."""

    @staticmethod
    def text_transform(params: dict, context: dict) -> dict:
        text = params.get("text") or context.get("last_output", "")
        op = params.get("operation", "uppercase")
        ops = {
            "uppercase": lambda t: t.upper(),
            "lowercase": lambda t: t.lower(),
            "title_case": lambda t: t.title(),
            "strip": lambda t: t.strip(),
            "reverse": lambda t: t[::-1],
            "word_count": lambda t: {"words": len(t.split()), "chars": len(t)},
        }
        fn = ops.get(op, lambda t: t)
        return {"result": fn(text), "operation": op}

    @staticmethod
    def data_filter(params: dict, context: dict) -> dict:
        data = params.get("data") or context.get("last_output", [])
        if not isinstance(data, list):
            return {"error": "data must be a list", "data": data}
        key = params.get("key")
        value = params.get("value")
        op = params.get("operator", "equals")
        if key and value is not None:
            ops_map = {
                "equals": lambda x: str(x.get(key, "")) == str(value),
                "contains": lambda x: str(value).lower() in str(x.get(key, "")).lower(),
                "greater_than": lambda x: float(x.get(key, 0)) > float(value),
                "less_than": lambda x: float(x.get(key, 0)) < float(value),
                "not_equals": lambda x: str(x.get(key, "")) != str(value),
            }
            fn = ops_map.get(op, ops_map["equals"])
            filtered = [item for item in data if fn(item)]
        else:
            filtered = data
        return {"filtered": filtered, "original_count": len(data),
                "result_count": len(filtered)}

    @staticmethod
    def data_transform(params: dict, context: dict) -> dict:
        data = params.get("data") or context.get("last_output", [])
        op = params.get("operation", "extract_field")
        if op == "extract_field" and isinstance(data, list):
            field = params.get("field", "id")
            result = [item.get(field) for item in data if isinstance(item, dict)]
            return {"result": result, "field": field}
        if op == "count":
            return {"count": len(data) if isinstance(data, (list, dict, str)) else 1}
        if op == "unique" and isinstance(data, list):
            return {"result": list(set(str(x) for x in data))}
        if op == "flatten" and isinstance(data, list):
            flat = []
            for item in data:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)
            return {"result": flat}
        return {"result": data, "operation": op}

    @staticmethod
    def conditional(params: dict, context: dict) -> dict:
        condition = params.get("condition", "true")
        true_action = params.get("true_action", "continue")
        false_action = params.get("false_action", "skip")
        last = context.get("last_output")
        if condition == "not_empty":
            result = bool(last)
        elif condition == "is_list":
            result = isinstance(last, list)
        elif condition == "has_error":
            result = isinstance(last, dict) and "error" in last
        elif condition == "true":
            result = True
        elif condition == "false":
            result = False
        else:
            result = bool(last)
        return {"condition": condition, "result": result,
                "action": true_action if result else false_action}

    @staticmethod
    def format_output(params: dict, context: dict) -> dict:
        data = params.get("data") or context.get("last_output")
        fmt = params.get("format", "json")
        template = params.get("template", "")
        if fmt == "json":
            try:
                return {"formatted": json.dumps(data, indent=2, default=str)}
            except Exception:
                return {"formatted": str(data)}
        if fmt == "csv" and isinstance(data, list) and data:
            if isinstance(data[0], dict):
                headers = list(data[0].keys())
                rows = [",".join(headers)]
                for item in data:
                    rows.append(",".join(str(item.get(h, "")) for h in headers))
                return {"formatted": "\n".join(rows)}
        if fmt == "template" and template:
            try:
                formatted = template.format(**context, data=data)
                return {"formatted": formatted}
            except Exception as e:
                return {"formatted": template, "error": str(e)}
        return {"formatted": str(data), "format": fmt}

    @staticmethod
    def http_request(params: dict, context: dict) -> dict:
        """Simulated HTTP request (for demo/dry-run)."""
        url = params.get("url", "")
        method = params.get("method", "GET")
        return {
            "simulated": True,
            "url": url, "method": method,
            "note": "HTTP requests are simulated in sandbox mode",
            "would_send": {"url": url, "method": method, "params": params.get("body", {})}
        }

    @staticmethod
    def log_message(params: dict, context: dict) -> dict:
        message = params.get("message", "Workflow step executed")
        level = params.get("level", "info")
        data = params.get("data") or context.get("last_output")
        log_fn = getattr(logger, level, logger.info)
        log_fn(f"[WORKFLOW LOG] {message}")
        return {"logged": True, "message": message, "level": level, "timestamp": datetime.utcnow().isoformat()}

    @staticmethod
    def wait(params: dict, context: dict) -> dict:
        ms = min(params.get("ms", 100), 5000)
        time.sleep(ms / 1000)
        return {"waited_ms": ms}

    @staticmethod
    def merge_context(params: dict, context: dict) -> dict:
        keys = params.get("keys", list(context.keys()))
        merged = {k: context[k] for k in keys if k in context}
        merged.update(params.get("extra", {}))
        return {"merged": merged}


ACTIONS = {
    "text_transform": ActionRegistry.text_transform,
    "data_filter": ActionRegistry.data_filter,
    "data_transform": ActionRegistry.data_transform,
    "conditional": ActionRegistry.conditional,
    "format_output": ActionRegistry.format_output,
    "http_request": ActionRegistry.http_request,
    "log_message": ActionRegistry.log_message,
    "wait": ActionRegistry.wait,
    "merge_context": ActionRegistry.merge_context,
}


class AutomationService:
    def __init__(self):
        self.openai = OpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
        self._workflows: dict[str, WorkflowRecord] = {}

    # ── LLM-based Workflow Generation ────────────────────────────────────
    def generate_workflow(self, description: str, context_data: dict = None) -> dict:
        available_actions = list(ACTIONS.keys()) + [
            "llm_complete", "nlp_sentiment", "nlp_ner", "nlp_summarise",
            "nlp_keywords", "genai_blog", "genai_email", "genai_code"
        ]
        prompt = f"""Generate a structured automation workflow for this task.

TASK: {description}
CONTEXT DATA: {json.dumps(context_data or {}, indent=2)}
AVAILABLE ACTIONS: {available_actions}

Return a JSON object with this exact structure:
{{
  "name": "Workflow name",
  "description": "What this workflow does",
  "steps": [
    {{
      "step_id": 1,
      "action": "action_name",
      "description": "What this step does",
      "params": {{}},
      "depends_on": []
    }}
  ],
  "expected_output": "What the workflow will produce"
}}

Important: Only use actions from the AVAILABLE ACTIONS list.
Return ONLY the JSON, no markdown."""

        if not self.openai:
            return self._demo_workflow(description)
        t0 = time.time()
        resp = self.openai.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert automation workflow designer. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, max_tokens=2000,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = re.sub(r'```\w*\n?', '', raw).replace('```', '').strip()
        try:
            workflow_def = json.loads(raw)
        except json.JSONDecodeError:
            return self._demo_workflow(description)
        return workflow_def | {"generated": True, "latency_ms": round((time.time() - t0) * 1000)}

    def _demo_workflow(self, description: str) -> dict:
        return {
            "name": "Demo Workflow",
            "description": description,
            "steps": [
                {"step_id": 1, "action": "log_message", "description": "Start workflow",
                 "params": {"message": f"Starting: {description}", "level": "info"}, "depends_on": []},
                {"step_id": 2, "action": "text_transform", "description": "Process input",
                 "params": {"text": description, "operation": "word_count"}, "depends_on": [1]},
                {"step_id": 3, "action": "format_output", "description": "Format results",
                 "params": {"format": "json"}, "depends_on": [2]},
            ],
            "expected_output": "Workflow execution trace",
            "generated": True, "demo_mode": True
        }

    # ── Workflow Execution ────────────────────────────────────────────────
    async def execute_workflow(self, workflow_def: dict,
                               initial_context: dict = None) -> WorkflowRecord:
        workflow_id = str(uuid.uuid4())[:8]
        steps_data = workflow_def.get("steps", [])
        steps = [
            WorkflowStep(
                step_id=s.get("step_id", i + 1),
                action=s.get("action", "log_message"),
                params=s.get("params", {}),
                description=s.get("description", ""),
                depends_on=s.get("depends_on", [])
            )
            for i, s in enumerate(steps_data)
        ]
        record = WorkflowRecord(workflow_id, workflow_def.get("name", "Unnamed"), steps)
        self._workflows[workflow_id] = record
        context = dict(initial_context or {})
        t0 = time.time()
        record.status = TaskStatus.RUNNING
        for step in steps:
            # Check dependencies
            dep_failed = any(
                next((s for s in steps if s.step_id == dep), None) and
                next((s for s in steps if s.step_id == dep)).status == TaskStatus.FAILED
                for dep in step.depends_on
            )
            if dep_failed:
                step.status = TaskStatus.SKIPPED
                logger.warning(f"Step {step.step_id} skipped — dependency failed")
                continue
            step.status = TaskStatus.RUNNING
            step.started_at = time.time()
            try:
                result = await self._execute_step(step, context)
                step.result = result
                step.status = TaskStatus.COMPLETED
                context["last_output"] = result
                context[f"step_{step.step_id}_output"] = result
            except Exception as e:
                step.error = str(e)
                step.status = TaskStatus.FAILED
                logger.error(f"Step {step.step_id} ({step.action}) failed: {e}")
                context["last_error"] = str(e)
            step.completed_at = time.time()
            await asyncio.sleep(0.01)
        failed = sum(1 for s in steps if s.status == TaskStatus.FAILED)
        record.status = TaskStatus.FAILED if failed > 0 else TaskStatus.COMPLETED
        record.completed_at = datetime.utcnow().isoformat()
        record.total_duration_ms = round((time.time() - t0) * 1000)
        return record

    async def _execute_step(self, step: WorkflowStep, context: dict) -> Any:
        action = step.action
        params = step.params
        if action in ACTIONS:
            return ACTIONS[action](params, context)
        # AI-powered actions
        if action == "llm_complete":
            from app.services.llm_service import LLMService
            svc = LLMService()
            prompt = params.get("prompt", str(context.get("last_output", "")))
            return svc.complete(prompt)
        if action == "nlp_sentiment":
            from app.services.nlp_service import NLPService
            svc = NLPService()
            text = params.get("text", str(context.get("last_output", "")))
            return svc.analyse_sentiment(text)
        if action == "nlp_ner":
            from app.services.nlp_service import NLPService
            svc = NLPService()
            text = params.get("text", str(context.get("last_output", "")))
            return svc.extract_entities(text)
        if action == "nlp_summarise":
            from app.services.nlp_service import NLPService
            svc = NLPService()
            text = params.get("text", str(context.get("last_output", "")))
            return svc.summarise(text)
        if action == "nlp_keywords":
            from app.services.nlp_service import NLPService
            svc = NLPService()
            text = params.get("text", str(context.get("last_output", "")))
            return svc.extract_keywords(text)
        if action == "genai_blog":
            from app.services.genai_service import GenAIService
            svc = GenAIService()
            return svc.generate_blog(params.get("topic", "AI trends"))
        if action == "genai_email":
            from app.services.genai_service import GenAIService
            svc = GenAIService()
            return svc.generate_email(params.get("purpose", "Follow up"))
        if action == "genai_code":
            from app.services.genai_service import GenAIService
            svc = GenAIService()
            return svc.generate_code(params.get("description", "Hello world"),
                                     params.get("language", "Python"))
        raise ValueError(f"Unknown action: {action}")

    # ── Workflow Management ───────────────────────────────────────────────
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowRecord]:
        return self._workflows.get(workflow_id)

    def list_workflows(self) -> list[dict]:
        return [
            {"workflow_id": wf.workflow_id, "name": wf.name, "status": wf.status,
             "step_count": len(wf.steps), "created_at": wf.created_at}
            for wf in self._workflows.values()
        ]

    # ── Workflow Templates ────────────────────────────────────────────────
    def list_templates(self) -> dict:
        return {
            "content_pipeline": {
                "name": "Content Generation Pipeline",
                "description": "Generate a blog post, extract keywords, then create social media posts",
                "steps": 4
            },
            "data_processing": {
                "name": "Data Processing Pipeline",
                "description": "Filter data, transform fields, format output",
                "steps": 3
            },
            "nlp_analysis": {
                "name": "Full NLP Analysis",
                "description": "Sentiment + NER + keywords + summary in sequence",
                "steps": 4
            },
            "document_intelligence": {
                "name": "Document Intelligence",
                "description": "Summarise document, extract entities, classify content, generate report",
                "steps": 4
            }
        }

    @property
    def available_actions(self) -> list[str]:
        return list(ACTIONS.keys()) + [
            "llm_complete", "nlp_sentiment", "nlp_ner",
            "nlp_summarise", "nlp_keywords",
            "genai_blog", "genai_email", "genai_code"
        ]
