from typing import Optional
from llm_optimizer import LLMOptimizer, OptimizationAttempt
import os

class GPUOptimizer:
    def __init__(self, baseline_code:str, kernel_name:str, model_provider: str ="lmstudio", model_name:str ="local_model", api_key:Optional[str] =None):
        """Args:
            baseline_code: Naive CUDA kernel
            kernel_name: Name of kernel function
            model_provider: LLM provider (lmstudio/openai)
            model_name: Model to use
            api_key: API key (or from env var)
        """
        self.baseline_code = baseline_code
        self.kernel_name =kernel_name

        self.llm_optimizer = LLMOptimizer(model_provider=model_provider, model_name=model_name, api_key= api_key or os.get_env("OPENAI_API"))
        self.baseline_time_ms =None
    
    