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

class ResearchObservation(BaseModel):
    done: bool = False
    reward: float = 0.0
    goal: str = ""
    rubric_count: int = 0
    attempt_number: int = 0
    criteria_met: int = 0
    feedback: str = ""
    revealed_hints: list[str] = Field(default_factory=list)

class ResearchState(BaseModel):
    episode_id: str = ""
    current_goal: str = ""
    rubric_count: int = 0
    attempt_number: int = 0
    best_score: float =0.0

class InternalState:
    def __init__(self, episode_id="", current_goal="", current_rubric=None, **kwargs):
        self.episode_id= episode_id
        self.current_goal = current_goal
        self.current_rubric = current_rubric or []
        self.attempt_number = 0
        self.best_score = 0.0

    def to_public_state(self)->ResearchState:
        return ResearchState(episode_id=self.episode_id, current_goal=self.current_goal, rubric_count=self.current_rubric, attempt_number=self.attempt_number, best_score=self.best_score)
    

