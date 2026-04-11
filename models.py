from pydantic import BaseModel, Field
from typing import Optional, Dict, Literal, List, Any


class Action(BaseModel):
    optimized_code: str
    strategy: Optional[str] = None
    expected_speedup: Optional[float] = None

class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    components: Dict[str, float]

class Observation(BaseModel):
    task_id: str
    task_name: str
    difficulty: Literal["easy", "medium", "hard"]
    baseline_code: str
    current_best_code: str
    current_best_speedup: float
    step_count: int
    max_steps: int
    pending_checks: List[str]
    completed_checks: List[str]
    done: bool

class EnvState(BaseModel):
    initialized: bool
    task_id: Optional[str] =None
    step_count: int = 0
    max_steps: int = 0
    total_reward: float = 0.0
    best_code: str = ""
    best_speedup: float = 1.0
    completed_checks: List[str] = Field(default_factory=list)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)

class ResetRequest(BaseModel):
    task_id: Optional[str] = None

class StepRequest(BaseModel):
    action: Action

class StepResult(BaseModel):
    observation:Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]

