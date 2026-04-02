from enum import Enum
from pydantic import BaseModel, Field


class ActionMode(str, Enum):
    SUBMIT = "submit"
    RESET ="reset"

class ResearchPlanAction(BaseModel):
    mode: ActionMode =Field(default=ActionMode.SUBMIT)
    research_plan: str = Field(default="")
    subset: str | None = None
    split: str | None = None

class ResearchObservation(BaseException):
    done: bool = False
    reward: float = 0.0
    goal: str = ""
    rubric_count: int = 0
    attempt_number: int = 0
    criteria_met: int = 0
    feedback: str = ""
    revealed_hints: list[str] = Field(default_factory=list)
