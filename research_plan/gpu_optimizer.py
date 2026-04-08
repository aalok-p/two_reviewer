from typing import Optional, Dict, Tuple
from llm_optimizer import LLMOptimizer, OptimizationAttempt
import os
import tempfile
import subprocess

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
    
    def compile_kernel(self, cuda_code:str) -> Tuple[bool, Optional[str], Optional[str]]:
        try:
            # write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(cuda_code)
                cu_file = f.name
            
            #cmpile with nvcc
            ptx_file = cu_file.replace('.cu', '.ptx')
            result = subprocess.run(
                ['nvcc', '--ptx', '-o', ptx_file, cu_file],
                capture_output=True,
                text=True,
                timeout=30,
            )
            os.unlink(cu_file)
            if result.returncode == 0:
                #read ptx
                with open(ptx_file, 'r') as f:
                    ptx_code = f.read()
                os.unlink(ptx_file)
                return True, ptx_code, None
            else:
                if os.path.exists(ptx_file):
                    os.unlink(ptx_file)
                return False, None, result.stderr
                
        except Exception as e:
            return False, None, str(e)
