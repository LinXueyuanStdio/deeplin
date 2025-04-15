from loguru import logger

from deeplin.inference_engine.base import InferenceEngine

try:
    from vllm import LLM, SamplingParams
except ImportError:
    logger.warning("VLLM not installed. Please install vllm with 'pip install vllm'.")



class vllmInferenceEngine(InferenceEngine):
    def __init__(self, model: str, max_tokens: int, tensor_parallel_size: int):
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_tokens,
            gpu_memory_utilization=0.95,
            enforce_eager=False,
        )

    def inference(self, prompts: list[str] | list[list[dict]], n=1, **kwargs) -> list[list[str]]:
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            max_tokens=kwargs.get("max_tokens", 1024),
            n=n,
        )
        outputs = self.llm.generate(prompts, sampling_params)
        responses = []
        for output in outputs:
            n_responses = []
            for i in range(n):
                n_responses.append(output.outputs[i].text)
            responses.append(n_responses)
        return responses
