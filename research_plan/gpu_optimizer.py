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

    def profile_kernel(self, ptx_code:str, inputs:Dict)->Dict:
        return {'execution_time_ms': 0.0, 'occupancy': 0.0, 'memory_efficiency': 0.0, 'warp_divergence': 0.0,'shared_memory_bytes': 0,'registers_per_thread': 0,}
    
    def optimize(self, input_generator, validator, max_iterations: int=10, target_speedup: float=10.0) ->Tuple[str, float]:
        print("llm powered kernel optimization")
        print(f"Model: {self.llm_optimizer.model_name}")
        print(f"Max iterations: {max_iterations}")
        print(f"Target speedup: {target_speedup}×\n")

        print("compiling baseline kernel...")
        baseline_success, baseline_ptx, baseline_error = self.compile_kernel(self.baseline_code)
        
        if not baseline_success:
            print(f"baseline compilation failed: {baseline_error}")
            return self.baseline_code, 1.0
        
        print("baseline compiled successfully")
        
        #pofile baseline
        inputs =input_generator()
        baseline_metrics = self.profile_kernel(baseline_ptx, inputs)
        self.baseline_time_ms = baseline_metrics.get('execution_time_ms', 1.0)
        
        if self.baseline_time_ms == 0.0:
            self.baseline_time_ms = 1.0 
        
        print(f"Baseline time: {self.baseline_time_ms:.3f}ms\n")

        