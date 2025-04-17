import requests
import os

from loguru import logger
from xlin import element_mapping

from deeplin.inference_engine.base import InferenceEngine


def get_userid_and_token(
    url="https://arsenal-openai.10jqka.com.cn:8443/vtuber/auth/api/oauth/v1/login",
    app_id="2d59cf942fc94140bace05cd97ccde09",
    app_secret="KIqlaKZypBDf7fcvIhT9g37v2wmQCSCjR3nlciqAlvg=",
):
    d = {"app_id": app_id, "app_secret": app_secret}
    h = {"Content-Type": "application/json"}
    r = requests.post(url, json=d, headers=h)
    data = r.json()["data"]
    return data["user_id"], data["token"]


def api_inference(
    user_id: str,
    token: str,
    input_message: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    n: int,
    timeout: int,
    multi_modal: bool = False,
    debug: bool = False,
):
    chat_h = {"Content-Type": "application/json", "userId": user_id, "token": token}
    params = {
        "messages": input_message,
        "temperature": temperature,
        "model": model,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "n": n,
    }
    version = "v1"
    if model == "claude":
        symbol = "claude"
        params["model"] = "claude-3-7-sonnet@20250219"
        params['anthropic_version'] = 'vertex-2023-10-16'
        version = "v3"
    elif "doubao" in model or model in ["ep-20250204210426-gclbn", "ep-20250410151344-fzm9z", "deepseek-reasoner", "deepseek-chat"]:
        symbol = "doubao"
        version = "v3"
        if "r1" in model or "reasoner" in model:
            params["model"] = 'ep-20250410145517-rpbrz'
            del params["n"] # remove n for reasoning
        elif "v3" in model or "chat" in model:
            params["model"] = 'ep-20250410151344-fzm9z'
    elif model == "r1-qianfan":
        symbol = "qianfan"
        params["model"] = 'deepseek-r1'
        del params["n"] # remove n for reasoning
    elif model == "gemini":
        symbol = "gemini"
        params["model"] = 'gemini-2.5-pro-preview-03-25'
    else:
        symbol = "chatgpt"

    if multi_modal:
        chat_url = 'https://arsenal-openai.10jqka.com.cn:8443/vtuber/ai_access/chatgpt/v1/picture/chat/completions'
    else:
        chat_url = f'https://arsenal-openai.10jqka.com.cn:8443/vtuber/ai_access/{symbol}/{version}/chat/completions'

    res = requests.post(
        chat_url,
        json=params,
        headers=chat_h,
        timeout=timeout,
    )
    resp = res.json()
    if debug or "choices" not in resp:
        logger.debug(resp)
    choices = resp.get("choices", [])
    responses: list[str] = []
    if len(choices) == 0:
        logger.warning("No response from server.")
        return [None] * n
    if len(choices) < n:
        logger.warning(f"Expected {n} responses, but got {len(choices)}.")
    for i in range(min(n, len(choices))):
        item = choices[i]
        if "message" in item:
            message = item["message"]
            content = message["content"] if "content" in message else ""
            reasoning_content = message["reasoning_content"] if "reasoning_content" in message else ""
            if len(reasoning_content) > 0:
                content = f"<think>\n{reasoning_content}\n</think>\n{content}"
        elif "text" in item:
            content = item["text"]
        else:
            content = item
        responses.append(content)
    if len(responses) < n:
        responses += [None] * (n - len(responses))
    return responses

class ApiInferenceEngine(InferenceEngine):
    def __init__(self, model: str, max_tokens: int, temperature: float = 0.6, top_p: float = 1.0):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        app_id = os.getenv("HITHINK_APP_ID")
        app_secret = os.getenv("HITHINK_APP_SECRET")
        if app_id is None or app_secret is None:
            raise ValueError("HITHINK_APP_ID and HITHINK_APP_SECRET must be set in environment variables.")
        self.user_id, self.token = get_userid_and_token(app_id=app_id, app_secret=app_secret)
        logger.debug(f"User ID: {self.user_id}, Token: {self.token}")
        available_models = [
            "gpt-3.5-turbo",
            "gpt4o",
            "gpt4",
            "claude",
            "gemini",
            "doubao-deepseek-r1", "ep-20250204210426-gclbn", "deepseek-reasoner", # deepseek-reasoner
            "doubao-deepseek-v3", "ep-20250410145517-rpbrz", "deepseek-chat", # deepseek-chat
            "r1-qianfan",
        ]
        if model not in available_models:
            logger.warning(f"Model {model} is not available. Please choose from {available_models}.")

    def inference(self, prompts: list[str] | list[list[dict]], n=1, **kwargs) -> list[list[str]]:
        model = kwargs.get("model", self.model)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        timeout = kwargs.get("timeout", 100)
        debug = kwargs.get("debug", False)
        messages_list = []
        for prompt in prompts:
            if isinstance(prompt, dict):
                messages_list.append([prompt])
            elif isinstance(prompt, str):
                messages_list.append([{"role": "user", "content": prompt}])
            elif isinstance(prompt, list):
                messages_list.append(prompt)
            else:
                raise ValueError(f"Invalid prompt format: {prompt}")

        def f(messages: list[dict]):
            return True, api_inference(
                self.user_id,
                self.token,
                messages,
                model,
                max_tokens,
                temperature,
                top_p,
                n,
                timeout,
                debug,
            )
        if len(messages_list) == 1:
            return f(messages_list[0])[1]
        responses = element_mapping(messages_list, f)
        return responses
