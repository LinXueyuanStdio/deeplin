import argparse
from pathlib import Path
from functools import partial

from dotenv import load_dotenv
from loguru import logger

from xlin import read_as_dataframe, dataframe_to_json_list, ls, xmap
from deeplin.inference_engine import build_inference_engine, batch_inference


load_dotenv()


def main(args):
    """
    1. read data: [{"prompt": "<|im_start|>user\n\nprompt", "messages": [{"role": "user", "content": "prompt"}]}]
    2. build inference engine
    3. run inference
    4. save results at key 'choices': [{"index": 0, "message": {"role": "assistant", "content": "<think> reasoning process here </think><answer> answer here </answer>"}}]
    """
    batch_size = args.batch_size
    inference_engine = build_inference_engine(
        engine=args.engine,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    data_paths = ls(args.data_dir)
    output_dir = Path(args.save_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    f = partial(
        batch_inference,
        inference_engine=inference_engine,
        prompt_key=args.prompt_key,
        n=args.n,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    for path in data_paths:
        save_path = output_dir / path.with_suffix(".jsonl")
        df = read_as_dataframe(path)
        jsonlist = dataframe_to_json_list(df)
        if not jsonlist:
            logger.warning(f"Empty json list for {path}.")
            continue
        logger.info(
            f"Loaded dataset from {path}, {len(df)} records. Cache path: {save_path}"
        )
        xmap(
            jsonlist,
            f,
            output_path=save_path,
            force_overwrite=False,
            batch_size=batch_size,
            is_batch_work_func=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--save_dir", type=str, default="output", help="Save directory")
    parser.add_argument(
        "--engine",
        type=str,
        choices=["vllm", "api", "openai"],
        default="vllm",
        help="Inference engine to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path. r1 is ep-20250204210426-gclbn",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=8192, help="Maximum number of tokens"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for VLLM")
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for VLLM",
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Number of responses to generate"
    )
    parser.add_argument(
        "--timeout", type=int, default=100, help="Timeout for API requests"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="prompt",
        help="Key for the prompt in the input data",
    )

    args = parser.parse_args()

    main(args)
