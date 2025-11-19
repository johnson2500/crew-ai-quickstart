# Llama Stack Multi-Agent Workflow Examples with CrewAI

This project demonstrates different approaches to building multi-agent workflows using **Llama Stack** and **CrewAI**. It provides examples of how to create agentic systems that can use tools, coordinate multiple agents, and leverage both frameworks together.

# Working with the UI

The UI is comprised of the following:

1) Agent Creation & Task Creation
2) Local RAG Chatbot Setup.
3) Launch Agents



## 📁 Project Structure

### Core Files

- **`main.py`** - Basic Llama Stack Agent examples
  - Demonstrates using the native `llama_stack_client.Agent` class
  - Shows how to use built-in tools like `web_search` and `file_search` (RAG)
  - Examples of session management and turn-based conversations
  - Vector store registration example

- **`multi_agent.py`** - Custom function calling with Llama Stack
  - Implements a manual agentic loop using `chat.completions` API
  - Custom tool registry with local functions (weather, activities, news)
  - Demonstrates function calling pattern with tool execution loop
  - Uses raw `llama_stack_client` for chat completions

- **`multi_agent_crewai.py`** - CrewAI multi-agent orchestration
  - Uses CrewAI framework for agent coordination
  - Multiple specialized agents (Weather, Activity, News, Coordinator)
  - Task-based workflow with dependencies
  - Traditional CrewAI pattern with agents working on tasks

- **`multi_agent_with_agent_tools.py`** - Agents as Tools pattern
  - Advanced pattern where agents are wrapped as tools
  - Main coordinator agent uses specialized agent tools
  - Each agent tool internally runs its own crew
  - Demonstrates hierarchical agent architecture

### Supporting Files

- **`llama_stack_agents/LlamaStackLLM.py`** - Custom LLM wrapper for CrewAI
  - Implements `crewai.BaseLLM` interface
  - Allows CrewAI to use Llama Stack as the LLM backend
  - Handles message formatting and tool integration
  - Used in `multi_agent_with_agent_tools.py`

- **`start_llama_stack.sh`** - Startup script
  - Starts Ollama in background
  - Syncs project dependencies with `uv`
  - Launches Llama Stack server on port 8321

- **`pyproject.toml`** - Project dependencies
  - Llama Stack and client libraries
  - CrewAI with tools support
  - Various ML/AI dependencies

## 🚀 Quick Start

### Prerequisites

1. **Ollama** installed and running
2. **Llama Stack** dependencies installed
3. **Python 3.10+**

### Setup

1. **Start Llama Stack server:**
   ```bash
   ./start_llama_stack.sh
   ```
   This will:
   - Start Ollama with `llama3.2:3b` model
   - Install dependencies
   - Launch Llama Stack on `http://localhost:8321`

2. **Install Python dependencies:**
   ```bash
   uv sync
   # or
   pip install -e .
   ```

3. **Run examples:**
   ```bash
   # Basic Llama Stack agent
   python main.py
   
   # Custom function calling
   python multi_agent.py
   
   # CrewAI multi-agent
   python multi_agent_crewai.py
   
   # Agents as tools pattern
   python multi_agent_with_agent_tools.py
   ```

## 🔄 Llama Stack vs CrewAI

### Llama Stack Agents

**Location:** `main.py`, `multi_agent.py`

**Characteristics:**
- Native agent implementation from `llama_stack_client`
- Direct integration with Llama Stack server
- Built-in tools: `web_search`, `file_search` (RAG), `function` tools
- Session-based conversations
- Turn-based execution with streaming support
- Lower-level control over agent behavior

**Use Cases:**
- When you need direct control over the agent loop
- When using Llama Stack's built-in tools (web search, RAG)
- When you want to integrate with Llama Stack's native features
- For simpler, single-agent workflows

**Example:**
```python
from llama_stack_client import Agent, LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321")
agent = Agent(
    client,
    model="ollama/llama3.2:3b",
    instructions="You are a helpful assistant.",
    tools=[{"type": "web_search"}]
)
```

### CrewAI Agents

**Location:** `multi_agent_crewai.py`, `multi_agent_with_agent_tools.py`

**Characteristics:**
- High-level agent orchestration framework
- Role-based agents with goals and backstories
- Task-based workflow with dependencies
- Automatic task orchestration and delegation
- Built-in tool system with `BaseTool` classes
- Better for complex multi-agent scenarios

**Use Cases:**
- When you need multiple agents working together
- For structured, role-based agent systems
- When you want automatic task orchestration
- For complex workflows with dependencies
- When you need agents to delegate to each other

**Example:**
```python
from crewai import Agent, Task, Crew

weather_agent = Agent(
    role="Weather Specialist",
    goal="Get accurate weather information",
    tools=[GetWeatherTool()],
)

crew = Crew(agents=[weather_agent], tasks=[weather_task])
result = crew.kickoff()
```

## 🛠️ Tools and Agents

### Base Tools

All examples use these base tools:

1. **`GetWeatherTool`** - Gets weather for a location
2. **`SuggestActivityTool`** - Suggests activities based on weather
3. **`GetNewsTool`** - Fetches and summarizes news

### Agent Types

#### 1. **Weather Specialist Agent**
- Role: Get accurate weather information
- Tools: `GetWeatherTool`
- Used in: CrewAI examples

#### 2. **Activity Planner Agent**
- Role: Suggest outdoor activities
- Tools: `SuggestActivityTool`
- Depends on: Weather information

#### 3. **News Reporter Agent**
- Role: Fetch and summarize news
- Tools: `GetNewsTool`
- Independent agent

#### 4. **Information Coordinator Agent**
- Role: Synthesize information from multiple sources
- Tools: Varies by example
- Coordinates: All other agents

## 📊 Architecture Patterns

### Pattern 1: Manual Agentic Loop
**File:** `multi_agent.py`

- Manual message loop with tool execution
- Direct function calling via `chat.completions`
- Full control over conversation flow
- Good for learning and debugging

### Pattern 2: CrewAI Task Orchestration
**File:** `multi_agent_crewai.py`

- Multiple agents with defined roles
- Tasks with dependencies
- Automatic orchestration by CrewAI
- Best for structured workflows

### Pattern 3: Agents as Tools
**File:** `multi_agent_with_agent_tools.py`

- Specialized agents wrapped as tools
- Main coordinator uses agent tools
- Hierarchical agent architecture
- Each agent tool runs its own crew internally
- Most flexible and composable pattern

## 🔌 Llama Stack Integration

### Using Llama Stack LLM with CrewAI

The project includes `LlamaStackLLM` class that allows CrewAI to use Llama Stack as the backend:

```python
from llama_stack_agents.LlamaStackLLM import LlamaStackLLM

llm = LlamaStackLLM(
    model="ollama/llama3.2:3b",
    endpoint="http://localhost:8321"
)

agent = Agent(
    role="...",
    llm=llm,  # Use Llama Stack instead of default LLM
    tools=[...]
)
```

### Llama Stack Built-in Tools

Llama Stack provides several built-in tools:

- **`web_search`** - Web search capabilities
- **`file_search`** - RAG with vector stores
- **`function`** - Custom function tools
- **`mcp`** - Model Context Protocol tools

## 🎯 Use Case Examples

### Example 1: Weather + Activity Query
```
"What's the weather in Boston and what's a good outdoor activity?"
```

**Flow:**
1. Weather agent gets weather → `{"temperature": "65°F", "conditions": "Sunny"}`
2. Activity agent uses weather → `{"activity": "walking in the Public Garden"}`
3. Coordinator synthesizes → Final answer

### Example 2: Multi-Source Information
```
"What's the weather, activity, and news in Boston?"
```

**Flow:**
1. Weather agent → Weather info
2. Activity agent → Activity suggestion (depends on weather)
3. News agent → News summary (parallel)
4. Coordinator → Comprehensive answer

## 🧪 Testing

Each example can be run independently:

```bash
# Test basic Llama Stack agent
python main.py

# Test function calling
python multi_agent.py

# Test CrewAI orchestration
python multi_agent_crewai.py

# Test agents as tools
python multi_agent_with_agent_tools.py
```

## 📝 Notes

### Model Configuration

All examples use `ollama/llama3.2:3b` by default. To use a different model:

1. Update the model string in each file
2. Ensure the model is available in your Llama Stack instance
3. Check available models: `llama-stack-client models list`

### Tool Execution

- Tools are executed synchronously
- Each tool returns a string (JSON for structured data)
- Tool results are added to conversation history
- Agents can call multiple tools in sequence

### Error Handling

- Network errors are caught and reported
- Unknown tools are logged and skipped
- Tool execution errors return error messages
- Agents continue execution even if some tools fail

## 🔗 Related Resources

- [Llama Stack Documentation](https://github.com/llama-stack/llama-stack)
- [CrewAI Documentation](https://docs.crewai.com/)
- [Llama Stack Client](https://github.com/llama-stack/llama-stack-client)

## 🤝 Contributing

This is an example project demonstrating different multi-agent patterns. Feel free to:
- Add more agent examples
- Create new tool types
- Experiment with different orchestration patterns
- Integrate additional Llama Stack features

## 📄 License

See individual file headers for license information.
