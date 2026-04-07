import os
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional, List

load_dotenv()

@dataclass
class OptimizationAttempt:
    iteration:int
    code: str
    compilation_success: bool
    compilation_error: Optional[str]
    correctness_passed: bool
    execution_time_ms: float
    speedup: float
    occupancy: float
    memory_efficiency: float
    reasoning: Optional[str] = None

class LLMOptimizer:
    def __init__(self, model_provider:Optional[str]=None, model_name:Optional[str]=None, api_key:Optional[str]=None):
        self.model_provder = self.model_provder or os.getenv("Model_Proivder", "lmstudio")

        if model_name is None:
            if self.model_provder == "lmstudio":
                self.model_name = os.getenv("LM_STUDIO_MODEL", "Qwen2.5 Coder 14B")
            elif self.model_provider =="openai":
                self.model_name = os.getenv("OPENAI_MODEL", "gpt-4")
            elif self.model_provider == "anthropic":
                self.model_name = os.getenv("ANTHROPIC_MODEL", "claude-opus")
            else:
                self.model_name = "local-model"
        else:
            self.model_name = model_name
        
        if api_key is None:
            if self.model_provider =="openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.model_provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            else:
                self.api_key = None 
        else:
            self.api_key = api_key
        self.history: List[OptimizationAttempt] = []