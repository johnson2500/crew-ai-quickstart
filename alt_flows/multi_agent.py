import json
import requests
from llama_stack_client import LlamaStackClient
import logging
# --- 1. Define Your Local Tools ---
# These are the functions your agent can "Act" with.
from shared import get_llama_stack_client, get_function_specifications, get_tool_registry

logger = logging.getLogger(__name__)

def run_agentic_workflow(user_prompt: str, model: str = "ollama/llama3.2:3b"):
    """
    Run an agentic workflow with tool calling.
    
    Args:
        user_prompt: The user's question or request
        model: The model identifier to use (check available models with `llama-stack-client models list`)
    """
    
    # Start the conversation history
    # Make the system message more explicit about using functions
    messages = [
        {"role": "system", "content": "You are a helpful assistant. When the user asks about weather or activities, you MUST use the get_weather, suggest_activity, and get_news functions. Do not provide general information - always call the functions to get accurate data."},
        {"role": "user", "content": user_prompt}
    ]
    
    logger.info(f"User Prompt: {user_prompt}\n")

    while True:
        # 1. Send the current state to the Llama Stack Server
        logger.info("...Agent sending state to server...")
        client = get_llama_stack_client()
        
        try:
            # Use the llama_stack_client to make the API call
            # Try "auto" first, but some models might need explicit function calling
            function_specifications = get_function_specifications()
            tool_registry = get_tool_registry()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                functions=function_specifications,
                function_call="auto"  # Let the model decide when to call functions
            )
            
            logger.debug(f"DEBUG: Response type: {type(response)}")
            logger.debug(f"DEBUG: Response choices: {len(response.choices) if response.choices else 0}")
            
            # Get the assistant's message from the response
            response_message_obj = response.choices[0].message
            
            # Debug: Print what we received
            logger.debug(f"DEBUG: Response message attributes: {dir(response_message_obj)}")
            logger.debug(f"DEBUG: Has tool_calls attr: {hasattr(response_message_obj, 'tool_calls')}")
            if hasattr(response_message_obj, 'tool_calls'):
                logger.debug(f"DEBUG: tool_calls value: {response_message_obj.tool_calls}")
            if hasattr(response_message_obj, 'function_call'):
                logger.debug(f"DEBUG: function_call value: {response_message_obj.function_call}")
            
            # Convert the message object to a dict for the messages list
            # The message might be a Pydantic model, so we convert it
            response_message = {
                "role": response_message_obj.role,
                "content": response_message_obj.content if hasattr(response_message_obj, 'content') else None,
            }
            
            # Check for both tool_calls (newer format) and function_call (older format)
            tool_calls_list = []
            
            # Try tool_calls first (OpenAI format)
            if hasattr(response_message_obj, 'tool_calls') and response_message_obj.tool_calls:
                for tc in response_message_obj.tool_calls:
                    if isinstance(tc, dict):
                        tool_calls_list.append(tc)
                    else:
                        tool_calls_list.append({
                            "id": tc.id,
                            "type": tc.type if hasattr(tc, 'type') else "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        })
            
            # Try function_call (older format, single function call)
            elif hasattr(response_message_obj, 'function_call') and response_message_obj.function_call:
                fc = response_message_obj.function_call
                if isinstance(fc, dict):
                    tool_calls_list.append({
                        "id": f"call_{len(tool_calls_list)}",
                        "type": "function",
                        "function": {
                            "name": fc.get("name", ""),
                            "arguments": fc.get("arguments", "{}")
                        }
                    })
                else:
                    tool_calls_list.append({
                        "id": f"call_{len(tool_calls_list)}",
                        "type": "function",
                        "function": {
                            "name": fc.name if hasattr(fc, 'name') else "",
                            "arguments": fc.arguments if hasattr(fc, 'arguments') else "{}"
                        }
                    })
            
            if tool_calls_list:
                response_message["tool_calls"] = tool_calls_list
            
        except Exception as e:
            logger.error(f"\n--- ERROR ---")
            logger.error(f"Could not connect to Llama Stack server at http://localhost:8321")
            logger.error(f"Please ensure the server is running.")
            logger.error(f"Error details: {e}")
            break
        
        # Add the server's response to our history
        messages.append(response_message)
        
        # 2. Check if the server gave a final answer OR called a tool
        if not response_message.get("tool_calls"):
            # The model gave a final text answer. Print it and exit the loop.
            logger.info(f"\nFinal Answer: {response_message.get('content', '')}")
            break
        
        # 3. If we are here, the server called a tool. We need to "Act".
        logger.info("...Server requested a tool call...")
        for tool_call in response_message["tool_calls"]:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            tool_call_id = tool_call["id"]
            
            function_to_call = tool_registry.get(function_name)
            
            if not function_to_call:
                logger.error(f"Error: Unknown tool '{function_name}'")
                continue
                
            tool_result = function_to_call(**function_args)
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": function_name,
                "content": tool_result
            })
            logger.info(f"--- AGENT Observation: {tool_result} ---")

run_agentic_workflow("What's the weather in Boston and what's a good outdoor activity and what's the news in Boston?")