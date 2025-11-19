from llama_stack_client import LlamaStackClient  # pyright: ignore[reportMissingImports]
import json
import requests  # pyright: ignore[reportMissingModuleSource]
import logging
logger = logging.getLogger(__name__)

client = LlamaStackClient(
    base_url="http://localhost:8321"
)

def get_llama_stack_client():
    return client

def get_weather(location: str) -> str:
    """Gets the current weather for a given location."""
    logger.info(f"--- AGENT EXECUTING: get_weather(location='{location}') ---")
    if location.lower() == "boston":
        return json.dumps({"temperature": "65°F", "conditions": "Sunny"})
    else:
        return json.dumps({"error": "Location not found"})

def suggest_activity(conditions: str, temp: str) -> str:
    """Suggests an outdoor activity based on weather conditions."""
    logger.info(f"--- AGENT EXECUTING: suggest_activity(conditions='{conditions}') ---")
    if conditions == "Sunny" and "65" in temp:
        return json.dumps({"activity": "walking in the Public Garden"})
    elif conditions == "Rainy":
        return json.dumps({"activity": "visiting the Museum of Fine Arts"})
    else:
        return json.dumps({"activity": "getting a coffee at a local cafe"})

def get_news(location: str) -> str:
    """Gets news based on location."""
    res = requests.get(f"https://www.boston.com")

    client = get_llama_stack_client()

    logger.info(f"--- AGENT EXECUTING: get_news(location='{location}') ---")
    response = client.chat.completions.create(
        model="ollama/llama3.2:3b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Given this repsonse from boston.com can you summarize the news in a few sentences? {res.text}"}
        ]
    )
    return response.choices[0].message.content


tool_registry = {
    "get_weather": get_weather,
    "suggest_activity": suggest_activity,
    "get_news": get_news,
}

def get_function_specifications() -> list:
    return [
        {
            "name": "get_weather",
            "description": "Gets the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., 'San Francisco, CA'",
                    },
                },
                "required": ["location"],
            },
        },
        {
            "name": "suggest_activity",
            "description": "Suggests an outdoor activity based on weather conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "conditions": {
                        "type": "string",
                        "description": "The weather condition, e.g., 'Sunny', 'Rainy'",
                    },
                    "temp": {
                        "type": "string",
                        "description": "The temperature, e.g., '65°F'",
                    }
                },
                "required": ["conditions", "temp"],
            },
        },
        {
            "name": "get_news",
            "description": "Gets news based on location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., 'Boston, MA'",
                    }
                },
                "required": ["location"],
            },
        }
    ]

def get_tool_registry():
    return tool_registry