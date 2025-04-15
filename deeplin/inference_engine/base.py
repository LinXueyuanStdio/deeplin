class InferenceEngine:
    def inference(self, prompts: list[str] | list[list[dict]], n=1, **kwargs) -> list[list[str]]:
        raise NotImplementedError("For each prompt, return n responses. If n=1, the return is a list of one element list of strings.")

    def inference_one(self, prompt: str | dict, **kwargs) -> list[str]:
        return self.inference([prompt], n=1, **kwargs)[0]

    def __call__(self, prompts: list[str] | list[list[dict]], n=1, **kwargs) -> list[list[str]]:
        return self.inference(prompts, n=n, **kwargs)
