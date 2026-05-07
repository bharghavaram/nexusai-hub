"""Automation API Routes — Workflow generation, execution, and management."""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
from app.services.automation_service import AutomationService

router = APIRouter(prefix="/automation", tags=["Automation Engine"])
_svc = None

def get_service() -> AutomationService:
    global _svc
    if _svc is None:
        _svc = AutomationService()
    return _svc


class GenerateWorkflowRequest(BaseModel):
    description: str = Field(..., min_length=10,
        description="Plain English description of what the workflow should do")
    context_data: Optional[dict] = Field(None, description="Initial data context for the workflow")

class ExecuteWorkflowRequest(BaseModel):
    workflow_def: dict = Field(..., description="Workflow definition (from /generate or custom)")
    initial_context: Optional[dict] = Field(None, description="Initial context variables")

class RunDescriptionRequest(BaseModel):
    description: str = Field(..., min_length=10,
        description="Describe the workflow in plain English — it will be generated and executed immediately")
    context_data: Optional[dict] = Field(None)

class StepRequest(BaseModel):
    action: str = Field(..., description="Action to execute")
    params: dict = Field(default_factory=dict)
    context: Optional[dict] = Field(default_factory=dict)


@router.get("/health")
async def health():
    svc = get_service()
    return {
        "status": "ok",
        "total_workflows": len(svc.list_workflows()),
        "available_actions": len(svc.available_actions),
        "llm_configured": bool(svc.openai),
    }

@router.post("/generate")
async def generate_workflow(req: GenerateWorkflowRequest):
    """Generate a workflow definition from a plain English description."""
    try:
        return get_service().generate_workflow(req.description, req.context_data)
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/execute")
async def execute_workflow(req: ExecuteWorkflowRequest):
    """Execute a workflow definition and return the full execution trace."""
    try:
        import asyncio
        svc = get_service()
        record = await svc.execute_workflow(req.workflow_def, req.initial_context)
        return record.to_dict()
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/run")
async def generate_and_run(req: RunDescriptionRequest):
    """One-shot: describe in English → generate workflow → execute immediately."""
    svc = get_service()
    try:
        workflow_def = svc.generate_workflow(req.description, req.context_data)
        record = await svc.execute_workflow(workflow_def, req.context_data)
        return {
            "workflow_def": workflow_def,
            "execution": record.to_dict(),
            "success": record.status == "completed"
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/step/execute")
async def execute_single_step(req: StepRequest):
    """Execute a single automation step for testing."""
    from app.services.automation_service import ACTIONS, WorkflowStep
    if req.action not in get_service().available_actions:
        raise HTTPException(400, f"Unknown action '{req.action}'. Available: {get_service().available_actions}")
    if req.action in ACTIONS:
        try:
            result = ACTIONS[req.action](req.params, req.context or {})
            return {"action": req.action, "result": result, "status": "completed"}
        except Exception as e:
            raise HTTPException(500, str(e))
    raise HTTPException(400, f"Action '{req.action}' requires workflow context")

@router.get("/workflows")
async def list_workflows():
    return {"workflows": get_service().list_workflows()}

@router.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    wf = get_service().get_workflow(workflow_id)
    if not wf:
        raise HTTPException(404, f"Workflow '{workflow_id}' not found")
    return wf.to_dict()

@router.get("/actions")
async def list_actions():
    return {"actions": get_service().available_actions,
            "total": len(get_service().available_actions)}

@router.get("/templates")
async def list_templates():
    return {"templates": get_service().list_templates()}
