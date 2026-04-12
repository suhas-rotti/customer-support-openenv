"""
FastAPI service for SmartCustomerSupportEnv.

This server exposes simple JSON endpoints for OpenEnv-style validation and
deployment on Hugging Face.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from env import SmartCustomerSupportEnv


class StepRequest(BaseModel):
    """Request body for evaluating one action."""

    action: str


class StepInfo(BaseModel):
    """Typed version of the environment info payload."""

    message: Optional[str] = None
    expected_keywords: Optional[Dict[str, List[str]]] = None
    matched_required: List[str] = Field(default_factory=list)
    missing_required: List[str] = Field(default_factory=list)
    matched_optional: List[str] = Field(default_factory=list)
    matched_empathy: List[str] = Field(default_factory=list)
    word_count: Optional[str] = None


class StepResponse(BaseModel):
    """Response returned by the step endpoint."""

    reward: float
    score: float
    done: bool
    info: StepInfo


class EvaluationResult(BaseModel):
    """Last evaluation result stored in the environment state."""

    reward: float
    score: float
    info: StepInfo


class ConversationTurn(BaseModel):
    """One customer or agent message in the state history."""

    role: str
    text: str


class StateResponse(BaseModel):
    """Typed state payload for the current environment."""

    level: Optional[str] = None
    scenario_id: Optional[str] = None
    customer_query: Optional[str] = None
    last_action: Optional[str] = None
    last_result: Optional[EvaluationResult] = None
    done: bool
    history: List[ConversationTurn] = Field(default_factory=list)


app = FastAPI(
    title="Smart Customer Support Environment",
    description="OpenEnv-compatible API server for customer support tasks.",
    version="1.0.0",
)

_env = SmartCustomerSupportEnv(seed=42)
_lock = Lock()


def _clamp_api_reward(value: Any) -> float:
    safe_score = float(value)

    if safe_score <= 0.0:
        return 0.25
    if safe_score >= 1.0:
        return 0.75

    return max(0.25, min(safe_score, 0.70))


def _step_info_from_dict(data: Dict[str, Any]) -> StepInfo:
    """Convert the environment info dictionary into a typed model."""

    return StepInfo(
        message=data.get("message"),
        expected_keywords=data.get("expected_keywords"),
        matched_required=data.get("matched_required", []),
        missing_required=data.get("missing_required", []),
        matched_optional=data.get("matched_optional", []),
        matched_empathy=data.get("matched_empathy", []),
        word_count=data.get("word_count"),
    )


def _state_response_from_dict(data: Dict[str, Any]) -> StateResponse:
    """Convert env.state() output into a typed JSON response."""

    history = [
        ConversationTurn(role=turn["role"], text=turn["text"])
        for turn in data.get("history", [])
    ]

    last_result_data = data.get("last_result")
    last_result: Optional[EvaluationResult] = None
    if last_result_data is not None:
        safe_reward = _clamp_api_reward(last_result_data["reward"])
        last_result = EvaluationResult(
            reward=safe_reward,
            score=safe_reward,
            info=_step_info_from_dict(last_result_data["info"]),
        )

    return StateResponse(
        level=data.get("level"),
        scenario_id=data.get("scenario_id"),
        customer_query=data.get("customer_query"),
        last_action=data.get("last_action"),
        last_result=last_result,
        done=bool(data.get("done", False)),
        history=history,
    )


@app.get("/")
def root() -> Dict[str, Any]:
    """Health and metadata endpoint."""

    return {
        "status": "ok",
        "name": "smart_customer_support_system",
        "message": "Smart Customer Support Environment API is running.",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": {
            "reset": "GET /reset?level=<easy|medium|hard>",
            "step": "POST /step",
            "state": "GET /state",
        },
    }


@app.get("/reset")
@app.post("/reset")
def reset(level: str = Query("easy")) -> Dict[str, str]:
    """Start a new episode and return the observation."""

    with _lock:
        observation = _env.reset(level)

    return {"task": level, "observation": observation}


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    """Evaluate an action against the current episode."""

    try:
        with _lock:
            result = _env.step(request.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    reward = float(result.get("reward", 0.5))

    if reward <= 0.0:
        reward = 0.25
    elif reward >= 1.0:
        reward = 0.75

    reward = max(0.25, min(reward, 0.75))
    result["reward"] = reward

    return StepResponse(
        reward=result["reward"],
        score=result["reward"],
        done=bool(result.get("done", False)),
        info=_step_info_from_dict(result["info"]),
    )


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    """Return the current environment state."""

    with _lock:
        current_state = _env.state()
    return _state_response_from_dict(current_state)
