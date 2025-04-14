from deeplin.inference_engine.base import InferenceEngine


def build_inference_engine(args) -> InferenceEngine:
    """Build inference engine based on the provided arguments."""
    if args.engine == "openai":
        from .openai_engine import OpenAIApiInferenceEngine
        return OpenAIApiInferenceEngine(
            args.model,
            args.max_tokens,
            args.temperature,
            args.top_p,
        )
    elif args.engine == "vllm":
        from .vllm_engine import vllmInferenceEngine
        return vllmInferenceEngine(
            args.model,
            args.max_tokens,
            args.tensor_parallel_size,
        )
    elif args.engine == "api":
        from .hexin_engine import ApiInferenceEngine
        return ApiInferenceEngine(
            args.model,
            args.max_tokens,
            args.temperature,
        )
    else:
        raise ValueError(f"Unknown engine: {args.engine}")


def batch_inference(
    inference_engine: InferenceEngine,
    rows: list[dict],
    prompt_key: str,
    args,
):
    """Perform batch inference using the provided inference engine."""
    prompts = [row[prompt_key] for row in rows]
    responses = inference_engine.inference(
        prompts,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
    )
    for row, n_responses in zip(rows, responses):
        choices: list[dict] = row.get("choices", [])
        start_idx = len(choices)
        for i, response in enumerate(n_responses):
            response_message = {
                "index": start_idx + i,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                },
            }
            choices.append(response_message)
        row["choices"] = choices
    return rows
