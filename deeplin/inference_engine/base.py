class InferenceEngine:
    def inference(self, prompts: list[str] | list[list[dict]], n=1, **kwargs) -> list[list[str]] | list[str]:
        raise NotImplementedError("For each prompt, return n responses. If n=1, the return is a list of one element list of strings.")

    def inference_one(self, prompt: str | dict, **kwargs) -> list[str] | str:
        response = self.inference([prompt], n=1, **kwargs)
        return response[0] if isinstance(response, list) else response

    def __call__(self, prompts: list[str] | list[list[dict]], n=1, **kwargs) -> list[list[str]] | list[str]:
        response = self.inference(prompts, n=n, **kwargs)
        return response if isinstance(response, list) else [response]
