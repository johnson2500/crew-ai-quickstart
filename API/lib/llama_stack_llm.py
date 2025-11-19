import os
from crewai.llm import LLM  # pyright: ignore[reportMissingImports]
from llama_stack_client import LlamaStackClient  # pyright: ignore[reportMissingImports]

MODEL_NAME = os.getenv('LLAMA_STACK_MODEL_NAME') or "llama/llama-3-3-70b-instruct-w8a8"
# Priority: LLAMA_STACK_URL (set by Helm) > LLAMA_STACK_CLIENT_URL > fallback to service name
BASE_URL = "http://localhost:8321"

def to_str(value):
    if isinstance(value, str):
        return value
    if value is None:
        return ""

    return str(value)

# LLM Class Override for LlamaStack
# This is used to override the LLM class to use the LlamaStack client.
class LlamaStackLLM(LLM):
    _token_usage: dict[str, int] = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "successful_requests": 0,
        "cached_prompt_tokens": 0,
    }

    is_litellm: bool = False
    model_name: str = MODEL_NAME

    def __new__(cls, *args, **kwargs):
        return super(LLM, cls).__new__(cls)

    def __init__(self, 
        model_name=MODEL_NAME, 
        base_url=BASE_URL,
        is_litellm: bool = False
    ):
        # don't call LLM.__init__() — that triggers factory logic again.
        self.model_name = model_name
        self.base_url = base_url
        self.is_litellm = is_litellm

        self.client = LlamaStackClient(base_url=base_url)

    def call(self, prompt, **kwargs):
        """
        Calls the LlamaStack client to get a response from the LLM.
        This is used to override the LLM class to use the LlamaStack client.
        """
        system_prompt = to_str(prompt)
        user_input = to_str(kwargs.get("input", ""))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,   
            messages=messages
        )
        return response.choices[0].message.content
