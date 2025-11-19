import json
import os
import requests # type: ignore
from crewai import Agent, Task, Crew # type: ignore
from crewai.tools import BaseTool # type: ignore
from typing import Type
from pydantic import BaseModel, Field # type: ignore
from llama_stack_client import LlamaStackClient # type: ignore
from fast_api.lib.llama_stack_llm import LlamaStackLLM

# --- Initialize Llama Stack Client ---
client = LlamaStackClient(
    base_url="http://localhost:8321"
)

class GetWeatherInput(BaseModel):
    """Input schema for GetWeatherTool."""
    location: str = Field(..., description="The city and state, e.g., 'Boston, MA'")

class GetWeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Gets the current weather for a given location. Use this when the user asks about weather conditions."
    args_schema: Type[BaseModel] = GetWeatherInput

    def _run(self, location: str) -> str:
        """Execute the tool."""
        print(f"--- AGENT EXECUTING: get_weather(location='{location}') ---")
        if location.lower() == "boston":
            return json.dumps({"temperature": "65°F", "conditions": "Sunny"})
        else:
            return json.dumps({"error": "Location not found"})


class SuggestActivityInput(BaseModel):
    """Input schema for SuggestActivityTool."""
    conditions: str = Field(..., description="The weather condition, e.g., 'Sunny', 'Rainy'")
    temp: str = Field(..., description="The temperature, e.g., '65°F'")

class SuggestActivityTool(BaseTool):
    name: str = "suggest_activity"
    description: str = "Suggests an outdoor activity based on weather conditions. Use this after getting weather information."
    args_schema: Type[BaseModel] = SuggestActivityInput

    def _run(self, conditions: str, temp: str) -> str:
        """Execute the tool."""
        print(f"--- AGENT EXECUTING: suggest_activity(conditions='{conditions}', temp='{temp}') ---")
        if conditions == "Sunny" and "65" in temp:
            return json.dumps({"activity": "walking in the Public Garden"})
        elif conditions == "Rainy":
            return json.dumps({"activity": "visiting the Museum of Fine Arts"})
        else:
            return json.dumps({"activity": "getting a coffee at a local cafe"})


class GetNewsInput(BaseModel):
    """Input schema for GetNewsTool."""
    location: str = Field(..., description="The city and state, e.g., 'Boston, MA'")

class GetNewsTool(BaseTool):
    name: str = "get_news"
    description: str = "Gets news based on location. Use this when the user asks about news or current events in a location."
    args_schema: Type[BaseModel] = GetNewsInput

    def _run(self, location: str) -> str:
        """Execute the tool."""
        print(f"--- AGENT EXECUTING: get_news(location='{location}') ---")
        try:
            res = requests.get(f"https://www.boston.com", timeout=10)
            response = client.chat.completions.create(
                model="ollama/llama3.2:3b",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Given this response from boston.com can you summarize the news in a few sentences? {res.text[:2000]}"}  # Limit text length
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error fetching news: {str(e)}"


# --- 2. Create CrewAI Agents ---

weather_agent = Agent(
    role="Weather Specialist",
    goal="Get accurate weather information for any location requested by the user",
    backstory="You are an expert meteorologist who always uses the get_weather tool to provide accurate, real-time weather data. Never guess or provide general information.",
    tools=[GetWeatherTool()],
    verbose=True,
    allow_delegation=False,
    llm=LlamaStackLLM(model_name="ollama/llama3.2:3b")
)

activity_agent = Agent(
    role="Activity Planner",
    goal="Suggest appropriate outdoor activities based on current weather conditions",
    backstory="You are a local activity expert who knows the best things to do in any city based on weather. You always use the suggest_activity tool after receiving weather information.",
    tools=[SuggestActivityTool()],
    verbose=True,
    allow_delegation=False,
    llm=LlamaStackLLM(model_name="ollama/llama3.2:3b")
)

news_agent = Agent(
    role="News Reporter",
    goal="Get and summarize current news for any location",
    backstory="You are a professional news reporter who always uses the get_news tool to fetch and summarize current events. You provide concise, accurate news summaries.",
    tools=[GetNewsTool()],
    verbose=True,
    allow_delegation=False,
    llm=LlamaStackLLM(model_name="ollama/llama3.2:3b")
)

coordinator_agent = Agent(
    role="Information Coordinator",
    goal="Coordinate information from multiple sources and provide a comprehensive answer to the user",
    backstory="You are a coordinator who synthesizes information from weather specialists, activity planners, and news reporters to provide complete answers to user queries.",
    verbose=True,
    allow_delegation=True,  # Can delegate to other agents
    llm=LlamaStackLLM(model_name="ollama/llama3.2:3b")
)


# --- 3. Create Tasks ---

def create_tasks(user_query: str):
    """Create tasks based on the user query."""
    tasks = []
    weather_task = None
    activity_task = None
    news_task = None
    
    # Check what the user is asking for
    query_lower = user_query.lower()
    
    if "weather" in query_lower:
        weather_task = Task(
            description=f"Get the weather information requested in: {user_query}",
            agent=weather_agent,
            expected_output="JSON string with temperature and conditions",
        )
        tasks.append(weather_task)
    
    if "activity" in query_lower or "outdoor" in query_lower:
        # Activity task depends on weather task if it exists
        activity_context = [weather_task] if weather_task else []
        activity_task = Task(
            description=f"Based on the weather conditions, suggest an appropriate outdoor activity. User query: {user_query}",
            agent=activity_agent,
            expected_output="JSON string with activity suggestion",
            context=activity_context,
        )
        tasks.append(activity_task)
    
    if "news" in query_lower:
        news_task = Task(
            description=f"Get and summarize the news for the location mentioned in: {user_query}",
            agent=news_agent,
            expected_output="A concise summary of current news",
        )
        tasks.append(news_task)
    
    # Final coordination task depends on all previous tasks
    coordinator_task = Task(
        description=f"Based on all the information gathered, provide a comprehensive answer to the user's query: {user_query}",
        agent=coordinator_agent,
        expected_output="A well-structured response that addresses all aspects of the user's query",
        context=tasks.copy(),  # Depends on all previous tasks
    )
    tasks.append(coordinator_task)
    
    return tasks


# --- 4. Create and Run the Crew ---

def run_crewai_workflow(user_query: str):
    """
    Run a CrewAI workflow with multiple agents.
    
    Args:
        user_query: The user's question or request
    """
    print(f"\n{'='*60}")
    print(f"User Query: {user_query}")
    print(f"{'='*60}\n")
    
    # Create tasks based on the query
    tasks = create_tasks(user_query)
    
    # Create the crew
    crew = Crew(
        agents=[weather_agent, activity_agent, news_agent, coordinator_agent],
        tasks=tasks,
        verbose=True,
    )
    
    # Execute the crew
    result = crew.kickoff()
    
    print(f"\n{'='*60}")
    print("FINAL RESULT:")
    print(f"{'='*60}")
    print(result)
    
    return result


# --- Run the crew ---
if __name__ == "__main__":
    # Example query
    query = "What's the weather in Boston, what's a good outdoor activity, and what's the news in Boston?"
    run_crewai_workflow(query)

