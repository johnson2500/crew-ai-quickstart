from crewai import BaseLLM
from typing import Any, Dict, List, Optional, Union
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(
    base_url="http://localhost:8321"
)


class LlamaStackLLM(BaseLLM):
    def __init__(self, model: str, endpoint: str, **kwargs):
        super().__init__(model=model, **kwargs)
        self.endpoint = endpoint
    
    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,  # Accept but ignore
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[bool] = False,
        **kwargs: Any,
    ) -> Union[str, Any]:
        # Convert string to message format if needed
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Prepare payload WITHOUT callbacks
        payload = {
            "model": self.model,
            "messages": messages,
        }
        
        if tools:
            payload["tools"] = tools
        
        # Call llama-stack WITHOUT callbacks parameter
        response = client.chat.completions.create(**payload)
        
        return response.choices[0].message.content