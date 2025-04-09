class InferenceEngine:
    def inference(self, prompts: list[str] | list[list[dict]], n=1, **kwargs) -> list[list[str]]:
        raise NotImplementedError("For each prompt, return n responses. If n=1, the return is a list of one element list of strings.")
