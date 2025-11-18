import json
import requests
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from llama_stack_client import LlamaStackClient
from llama_stack_agents.LlamaStackLLM import LlamaStackLLM

# --- Initialize Llama Stack Client ---
client = LlamaStackClient(
    base_url="http://localhost:8321"
)


llm = LlamaStackLLM(
    model="ollama/llama3.2:3b",
    endpoint="http://localhost:8321"
)

# --- 1. Define Base Tools (for direct tool usage) ---

class GetWeatherInput(BaseModel):
    """Input schema for GetWeatherTool."""
    location: str = Field(..., description="The city and state, e.g., 'Boston, MA'")

class GetWeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Gets the current weather for a given location."
    args_schema: Type[BaseModel] = GetWeatherInput

    def _run(self, location: str) -> str:
        """Execute the tool."""
        print(f"--- TOOL EXECUTING: get_weather(location='{location}') ---")
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
    description: str = "Suggests an outdoor activity based on weather conditions."
    args_schema: Type[BaseModel] = SuggestActivityInput

    def _run(self, conditions: str, temp: str) -> str:
        """Execute the tool."""
        print(f"--- TOOL EXECUTING: suggest_activity(conditions='{conditions}', temp='{temp}') ---")
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
    description: str = "Gets news based on location."
    args_schema: Type[BaseModel] = GetNewsInput

    def _run(self, location: str) -> str:
        """Execute the tool."""
        print(f"--- TOOL EXECUTING: get_news(location='{location}') ---")
        try:
            res = requests.get(f"https://www.boston.com", timeout=10)
            response = client.chat.completions.create(
                model="ollama/llama3.2:3b",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Given this response from boston.com can you summarize the news in a few sentences? {res.text[:2000]}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error fetching news: {str(e)}"


# --- 2. Create Specialized Agent Tools (Agents that act as tools) ---

# Agent Tool: Weather Agent
weather_agent_tool_input = BaseModel

class WeatherAgentTool(BaseTool):
    """
    A tool that uses a specialized weather agent to get weather information.
    This agent can reason about weather queries and use multiple tools.
    """
    name: str = "weather_agent"
    description: str = (
        "A specialized agent that handles weather-related queries. "
        "Use this when you need weather information or weather-based recommendations. "
        "The agent will use appropriate tools to get accurate weather data."
    )
    args_schema: Type[BaseModel] = GetWeatherInput  # Reuse the same input schema

    def _run(self, location: str) -> str:
        """Execute the weather agent as a tool."""
        print(f"--- AGENT TOOL EXECUTING: weather_agent(location='{location}') ---")
        
        # Create a specialized weather agent
        weather_agent = Agent(
            role="Weather Specialist",
            goal="Get accurate weather information and provide weather-based insights",
            backstory="You are an expert meteorologist who always uses tools to get accurate weather data.",
            tools=[GetWeatherTool()],
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )
        
        # Create a task for the weather agent
        weather_task = Task(
            description=f"Get the current weather for {location} and provide a brief summary.",
            agent=weather_agent,
            expected_output="A summary of the current weather conditions",
        )
        
        # Run the agent as a crew
        weather_crew = Crew(
            agents=[weather_agent],
            tasks=[weather_task],
            verbose=False,  # Less verbose when used as a tool
        )
        
        result = weather_crew.kickoff()
        return str(result)


# Agent Tool: Activity Planning Agent
class ActivityAgentToolInput(BaseModel):
    """Input schema for ActivityAgentTool."""
    location: str = Field(..., description="The city and state, e.g., 'Boston, MA'")
    weather_info: str = Field(default="", description="Optional: Weather information if already available")

class ActivityAgentTool(BaseTool):
    """
    A tool that uses a specialized activity planning agent.
    This agent can suggest activities based on weather and location.
    """
    name: str = "activity_agent"
    description: str = (
        "A specialized agent that suggests outdoor activities based on location and weather. "
        "Use this when you need activity recommendations. "
        "The agent will first get weather information if needed, then suggest appropriate activities."
    )
    args_schema: Type[BaseModel] = ActivityAgentToolInput

    def _run(self, location: str, weather_info: str = "") -> str:
        """Execute the activity agent as a tool."""
        print(f"--- AGENT TOOL EXECUTING: activity_agent(location='{location}') ---")
        
        # Create specialized agents
        weather_agent = Agent(
            role="Weather Specialist",
            goal="Get weather information",
            backstory="You are a meteorologist who gets accurate weather data.",
            tools=[GetWeatherTool()],
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )
        
        activity_agent = Agent(
            role="Activity Planner",
            goal="Suggest appropriate outdoor activities based on weather",
            backstory="You are a local activity expert who suggests activities based on weather conditions.",
            tools=[SuggestActivityTool()],
            verbose=True,
            allow_delegation=False,
        )
        
        # Create tasks
        tasks = []
        
        # Get weather if not provided
        if not weather_info:
            weather_task = Task(
                description=f"Get the current weather for {location}",
                agent=weather_agent,
                expected_output="JSON string with temperature and conditions",
            )
            tasks.append(weather_task)
        
        # Suggest activity based on weather
        activity_task = Task(
            description=(
                f"Based on the weather information {'provided' if weather_info else 'gathered'}, "
                f"suggest an appropriate outdoor activity for {location}."
            ),
            agent=activity_agent,
            expected_output="A JSON string with activity suggestion",
            context=tasks if tasks else None,
        )
        tasks.append(activity_task)
        
        # Run the agents as a crew
        activity_crew = Crew(
            agents=[weather_agent, activity_agent],
            tasks=tasks,
            verbose=False,
        )
        
        result = activity_crew.kickoff()
        return str(result)


# Agent Tool: News Agent
class NewsAgentTool(BaseTool):
    """
    A tool that uses a specialized news agent to get and summarize news.
    """
    name: str = "news_agent"
    description: str = (
        "A specialized agent that fetches and summarizes news for a location. "
        "Use this when you need current news or events information."
    )
    args_schema: Type[BaseModel] = GetNewsInput

    def _run(self, location: str) -> str:
        """Execute the news agent as a tool."""
        print(f"--- AGENT TOOL EXECUTING: news_agent(location='{location}') ---")
        
        # Create a specialized news agent
        news_agent = Agent(
            role="News Reporter",
            goal="Get and summarize current news for any location",
            backstory="You are a professional news reporter who fetches and summarizes current events.",
            tools=[GetNewsTool()],
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )
        
        # Create a task for the news agent
        news_task = Task(
            description=f"Get and summarize the current news for {location}",
            agent=news_agent,
            expected_output="A concise summary of current news",
        )
        
        # Run the agent as a crew
        news_crew = Crew(
            agents=[news_agent],
            tasks=[news_task],
            verbose=False,
        )
        
        result = news_crew.kickoff()
        return str(result)


# --- 3. Create the Main Coordinator Agent (uses agent tools) ---

main_coordinator = Agent(
    role="Information Coordinator",
    goal="Coordinate information from multiple specialized agents and provide comprehensive answers",
    backstory=(
        "You are a coordinator who orchestrates specialized agents to answer complex queries. "
        "You have access to agent tools that can handle weather, activities, and news. "
        "Use these agent tools to gather information and synthesize comprehensive responses."
    ),
    tools=[
        WeatherAgentTool(),      # Agent as a tool
        ActivityAgentTool(),      # Agent as a tool
        NewsAgentTool(),          # Agent as a tool
    ],
    verbose=True,
    allow_delegation=False,  # The agent tools handle delegation internally
    llm=llm,
)


# --- 4. Create and Run the Main Crew ---

def run_agent_tools_workflow(user_query: str):
    """
    Run a workflow where the main agent uses other agents as tools.
    
    Args:
        user_query: The user's question or request
    """
    print(f"\n{'='*60}")
    print(f"User Query: {user_query}")
    print(f"{'='*60}\n")
    print("Using Agent Tools Pattern: Main agent will call specialized agent tools\n")
    
    # Create a task for the main coordinator
    coordinator_task = Task(
        description=(
            f"Answer the user's query: {user_query}\n"
            "Use the available agent tools (weather_agent, activity_agent, news_agent) "
            "to gather information and provide a comprehensive answer."
        ),
        agent=main_coordinator,
        expected_output="A comprehensive answer that addresses all aspects of the user's query",
    )
    
    # Create the crew with just the main coordinator
    crew = Crew(
        agents=[main_coordinator],
        tasks=[coordinator_task],
        verbose=True,
    )
    
    # Execute the crew
    result = crew.kickoff()
    
    print(f"\n{'='*60}")
    print("FINAL RESULT:")
    print(f"{'='*60}")
    print(result)
    
    return result


# --- Run the workflow ---
if __name__ == "__main__":
    # Example query
    query = "What's the weather in Boston, what's a good outdoor activity, and what's the latestnews in Boston?"
    run_agent_tools_workflow(query)

