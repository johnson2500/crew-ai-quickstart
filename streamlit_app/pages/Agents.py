import streamlit as st
import sys
import os
import json
import logging
from pathlib import Path

# Add parent directory to path to import alt_flows modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging to capture agent output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import API clients
try:
    from streamlit_app.lib.clients.crew_client import CrewClient
    from streamlit_app.lib.clients.task_client import TaskClient
    from streamlit_app.lib.clients.agent_client import AgentClient
    API_CLIENTS_AVAILABLE = True
except ImportError as e:
    API_CLIENTS_AVAILABLE = False
    logger.warning(f"API clients not available: {e}")

# Import agent workflows (with optional imports for dependencies)
try:
    from fast_api.alt_flows.multi_agent import run_agentic_workflow
    MULTI_AGENT_AVAILABLE = True
except ImportError as e:
    MULTI_AGENT_AVAILABLE = False
    logger.warning(f"Multi-agent workflow not available: {e}")

try:
    from fast_api.alt_flows.multi_agent_crewai import run_crewai_workflow
    CREWAI_AVAILABLE = True
except ImportError as e:
    CREWAI_AVAILABLE = False
    logger.warning(f"CrewAI workflow not available: {e}")

try:
    from fast_api.alt_flows.main import agent_1, agent_2
    LLAMA_STACK_AGENTS_AVAILABLE = True
except ImportError as e:
    LLAMA_STACK_AGENTS_AVAILABLE = False
    logger.warning(f"Llama Stack agents not available: {e}")

st.set_page_config(page_title="Agent Trigger", layout="wide")

# --- Session State ---
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "agent_results" not in st.session_state:
    st.session_state.agent_results = []
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = os.getenv("FASTAPI_BASE_URL", "http://localhost:8001")

def create_sidebar():
    """Create sidebar with agent selection and settings."""
    with st.sidebar:
        st.title("Agent Settings")
        
        # API Configuration
        if API_CLIENTS_AVAILABLE:
            st.subheader("API Configuration")
            api_base_url = st.text_input(
                "API Base URL",
                value=st.session_state.api_base_url,
                help="Base URL for the FastAPI backend"
            )
            st.session_state.api_base_url = api_base_url
            st.divider()
        
        # Build available agent types list
        available_agents = []
        if MULTI_AGENT_AVAILABLE:
            available_agents.append("Multi-Agent (Tool Calling)")
        if CREWAI_AVAILABLE:
            available_agents.append("CrewAI Multi-Agent")
        if LLAMA_STACK_AGENTS_AVAILABLE:
            available_agents.append("Llama Stack Agent (Basic)")
            available_agents.append("Llama Stack Agent (RAG)")
        
        if not available_agents:
            st.error("⚠️ No agent workflows available. Please install required dependencies.")
            st.info("Run: pip install crewai")
            return None, None
        
        agent_type = st.selectbox(
            "Select Agent Type",
            available_agents,
            index=0
        )
        
        model = st.selectbox(
            "Select Model",
            ["ollama/llama3.2:3b", "meta-llama/Llama-3.1-8B-Instruct"],
            index=0
        )
        
        st.divider()
        
        if st.button("Clear Agent History"):
            st.session_state.agent_messages = []
            st.session_state.agent_results = []
            st.rerun()
        
        st.info("💡 **Tip:** Use the Multi-Agent workflows to test tool calling and agent coordination.")
    
    return agent_type, model

def run_multi_agent_workflow(prompt: str, model: str):
    """Run the multi-agent workflow with tool calling."""
    if not MULTI_AGENT_AVAILABLE:
        return {
            "success": False,
            "error": "Multi-agent workflow is not available. Please check dependencies.",
            "result": None
        }
    try:
        # Capture output and logging
        import io
        from contextlib import redirect_stdout, redirect_stderr
        import logging
        
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        log_buffer = io.StringIO()
        
        # Set up a handler to capture logs
        handler = logging.StreamHandler(log_buffer)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to the logger used by the agent
        agent_logger = logging.getLogger('alt_flows.multi_agent')
        agent_logger.addHandler(handler)
        agent_logger.setLevel(logging.INFO)
        
        try:
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                run_agentic_workflow(prompt, model)
        finally:
            agent_logger.removeHandler(handler)
        
        stdout_output = output_buffer.getvalue()
        stderr_output = error_buffer.getvalue()
        log_output = log_buffer.getvalue()
        
        # Extract final answer from logs if available
        final_answer = None
        if "Final Answer:" in log_output:
            try:
                final_answer = log_output.split("Final Answer:")[-1].strip()
            except:
                pass
        
        combined_output = f"{stdout_output}\n{log_output}".strip()
        
        return {
            "success": True,
            "result": final_answer if final_answer else "Workflow completed. Check output log for details.",
            "stdout": combined_output,
            "stderr": stderr_output
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "result": None,
            "traceback": traceback.format_exc()
        }

def run_crewai_workflow_wrapper(prompt: str):
    """Run the CrewAI multi-agent workflow."""
    if not CREWAI_AVAILABLE:
        return {
            "success": False,
            "error": "CrewAI is not installed. Please run: pip install crewai",
            "result": None
        }
    try:
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            result = run_crewai_workflow(prompt)
        
        stdout_output = output_buffer.getvalue()
        stderr_output = error_buffer.getvalue()
        
        return {
            "success": True,
            "result": str(result) if result else "Workflow completed",
            "stdout": stdout_output,
            "stderr": stderr_output
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "result": None,
            "traceback": traceback.format_exc()
        }

def run_llama_stack_agent_basic():
    """Run the basic Llama Stack agent."""
    if not LLAMA_STACK_AGENTS_AVAILABLE:
        return {
            "success": False,
            "error": "Llama Stack agents are not available. Please check dependencies.",
            "result": None
        }
    try:
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            agent_1()
        
        stdout_output = output_buffer.getvalue()
        stderr_output = error_buffer.getvalue()
        
        return {
            "success": True,
            "result": "Basic agent workflow completed",
            "stdout": stdout_output,
            "stderr": stderr_output
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "result": None
        }

def run_llama_stack_agent_rag():
    """Run the RAG-enabled Llama Stack agent."""
    if not LLAMA_STACK_AGENTS_AVAILABLE:
        return {
            "success": False,
            "error": "Llama Stack agents are not available. Please check dependencies.",
            "result": None
        }
    try:
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            agent_2()
        
        stdout_output = output_buffer.getvalue()
        stderr_output = error_buffer.getvalue()
        
        return {
            "success": True,
            "result": "RAG agent workflow completed",
            "stdout": stdout_output,
            "stderr": stderr_output
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "result": None
        }

def create_task_management_ui():
    """Create UI for task and agent management using API clients."""
    if not API_CLIENTS_AVAILABLE:
        st.error("API clients are not available. Please check imports.")
        return
    
    st.title("📋 Task & Agent Management")
    st.markdown("Manage tasks and agents, and launch tasks using the API.")
    
    # Initialize clients
    try:
        task_client = TaskClient(base_url=st.session_state.api_base_url)
        agent_client = AgentClient(base_url=st.session_state.api_base_url)
        crew_client = CrewClient(base_url=st.session_state.api_base_url)
    except Exception as e:
        st.error(f"Failed to initialize API clients: {e}")
        st.info("Make sure the FastAPI server is running and the base URL is correct.")
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["View Tasks", "View Agents", "Launch Task"])
    
    # Tab 1: View Tasks
    with tab1:
        st.subheader("Available Tasks")
        try:
            tasks = task_client.get_tasks()
            if tasks:
                for task in tasks:
                    with st.expander(f"📝 {task.name} (ID: {task.id})"):
                        st.write(f"**Description:** {task.description}")
                        st.write(f"**Agent ID:** {task.agent_id}")
                        st.write(f"**Expected Output:** {task.expected_output}")
                        if task.dependencies:
                            st.write(f"**Dependencies:** {', '.join(task.dependencies)}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"Delete Task", key=f"delete_task_{task.id}"):
                                try:
                                    deleted_task = task_client.delete_task(task.id)
                                    st.success(f"Task '{deleted_task.name}' deleted successfully!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to delete task: {e}")
            else:
                st.info("No tasks available. Create a task in the 'Create Task' tab.")
        except Exception as e:
            st.error(f"Failed to fetch tasks: {e}")
            st.info("Make sure the FastAPI server is running.")
    
    # Tab 2: View Agents
    with tab2:
        st.subheader("Available Agents")
        try:
            agents = agent_client.get_agents()
            if agents:
                for agent in agents:
                    with st.expander(f"🤖 {agent.role} (ID: {agent.id})"):
                        st.write(f"**Goal:** {agent.goal}")
                        st.write(f"**Backstory:** {agent.backstory}")
                        if agent.tools:
                            st.write(f"**Tools:** {', '.join(agent.tools)}")
                        st.write(f"**Verbose:** {agent.verbose}")
                        st.write(f"**Allow Delegation:** {agent.allow_delegation}")
            else:
                st.info("No agents available. Create an agent in the 'Create Agent' page.")
        except Exception as e:
            st.error(f"Failed to fetch agents: {e}")
            st.info("Make sure the FastAPI server is running.")
    
    # Tab 3: Launch Task
    with tab3:
        st.subheader("Launch Task with Agent")
        try:
            # Get available tasks and agents
            tasks = task_client.get_tasks()
            agents = agent_client.get_agents()
            
            if not tasks:
                st.warning("No tasks available. Please create a task first.")
            elif not agents:
                st.warning("No agents available. Please create an agent first.")
            else:
                # Task selection
                task_options = {f"{task.name} (ID: {task.id})": task.id for task in tasks}
                selected_task_display = st.selectbox(
                    "Select Task",
                    options=list(task_options.keys())
                )
                selected_task_id = task_options[selected_task_display]
                
                # Agent selection
                agent_options = {f"{agent.role} (ID: {agent.id})": agent.id for agent in agents}
                selected_agent_display = st.selectbox(
                    "Select Agent",
                    options=list(agent_options.keys())
                )
                selected_agent_id = agent_options[selected_agent_display]
                
                # Display selected task and agent info
                selected_task = next(t for t in tasks if t.id == selected_task_id)
                selected_agent = next(a for a in agents if a.id == selected_agent_id)
                
                st.info(f"**Task:** {selected_task.name} - {selected_task.description}")
                st.info(f"**Agent:** {selected_agent.role} - {selected_agent.goal}")
                
                if st.button("🚀 Launch Task", type="primary"):
                    with st.spinner("Launching task..."):
                        try:
                            result = crew_client.launch_task_by_ids(
                                task_id=selected_task_id,
                                agent_id=selected_agent_id
                            )
                            
                            st.success("✅ Task launched successfully!")
                            st.subheader("Execution Result")
                            st.write(f"**Task ID:** {result.task_id}")
                            st.write(f"**Agent ID:** {result.agent_id}")
                            st.write(f"**Status:** {result.status}")
                            st.write(f"**Result:**")
                            st.code(result.result, language='text')
                            
                            # Store in session state
                            if "task_executions" not in st.session_state:
                                st.session_state.task_executions = []
                            st.session_state.task_executions.append({
                                "task_id": result.task_id,
                                "agent_id": result.agent_id,
                                "status": result.status,
                                "result": result.result
                            })
                        except Exception as e:
                            st.error(f"Failed to launch task: {e}")
                            import traceback
                            with st.expander("View Error Details"):
                                st.code(traceback.format_exc(), language='python')
        except Exception as e:
            st.error(f"Failed to load tasks/agents: {e}")
            st.info("Make sure the FastAPI server is running.")
            st.info("💡 **Tip:** To create new tasks, go to the 'Create Agent' page.")

def create_main_ui():
    """Create the main agent triggering UI."""
    st.title("🤖 Agent Trigger")
    st.markdown("Trigger and interact with various agent workflows.")
    
    # Get settings from sidebar
    agent_type, model = create_sidebar()
    
    # Check if sidebar returned None (no agents available)
    if agent_type is None or model is None:
        st.stop()
    
    # Display agent history
    if st.session_state.agent_messages:
        st.subheader("Agent Execution History")
        for i, msg in enumerate(st.session_state.agent_messages):
            with st.expander(f"Execution #{i+1}: {msg.get('agent_type', 'Unknown')} - {msg.get('prompt', '')[:50]}..."):
                st.write(f"**Agent Type:** {msg.get('agent_type', 'Unknown')}")
                st.write(f"**Prompt:** {msg.get('prompt', 'N/A')}")
                st.write(f"**Model:** {msg.get('model', 'N/A')}")
                
                if msg.get('result'):
                    st.write("**Result:**")
                    st.code(msg['result'], language='text')
                
                if msg.get('stdout'):
                    st.write("**Output:**")
                    st.code(msg['stdout'], language='text')
                
                if msg.get('error'):
                    st.error(f"**Error:** {msg['error']}")
                    if msg.get('traceback'):
                        with st.expander("View Traceback"):
                            st.code(msg['traceback'], language='python')
    
    # Agent input section
    st.divider()
    st.subheader("Trigger Agent")
    
    # Different input based on agent type
    if agent_type in ["Llama Stack Agent (Basic)", "Llama Stack Agent (RAG)"]:
        prompt = st.text_area(
            "Agent Configuration",
            value="These agents use predefined prompts. Click 'Run Agent' to execute.",
            height=100,
            disabled=True
        )
    else:
        prompt = st.text_area(
            "Enter your prompt or query:",
            placeholder="e.g., What's the weather in Boston and what's a good outdoor activity?",
            height=100
        )
    
    if st.button("🚀 Run Agent", type="primary"):
        if not prompt and agent_type not in ["Llama Stack Agent (Basic)", "Llama Stack Agent (RAG)"]:
            st.warning("Please enter a prompt before running the agent.")
            return
        
        # Show spinner while running
        with st.spinner(f"Running {agent_type}..."):
            result = None
            
            if agent_type == "Multi-Agent (Tool Calling)":
                result = run_multi_agent_workflow(prompt, model)
            elif agent_type == "CrewAI Multi-Agent":
                result = run_crewai_workflow_wrapper(prompt)
            elif agent_type == "Llama Stack Agent (Basic)":
                result = run_llama_stack_agent_basic()
            elif agent_type == "Llama Stack Agent (RAG)":
                result = run_llama_stack_agent_rag()
            
            # Store result in session state
            if result:
                agent_message = {
                    "agent_type": agent_type,
                    "prompt": prompt if prompt else "Predefined prompt",
                    "model": model if agent_type != "CrewAI Multi-Agent" else "CrewAI",
                    "result": result.get("result"),
                    "stdout": result.get("stdout"),
                    "stderr": result.get("stderr"),
                    "error": result.get("error"),
                    "traceback": result.get("traceback"),
                    "success": result.get("success", False)
                }
                st.session_state.agent_messages.append(agent_message)
        
        # Display result
        if result:
            if result.get("success"):
                st.success("✅ Agent execution completed!")
                
                if result.get("result"):
                    st.subheader("Result")
                    st.markdown(result["result"])
                
                if result.get("stdout"):
                    with st.expander("View Output Log"):
                        st.code(result["stdout"], language='text')
                
                if result.get("stderr"):
                    with st.expander("View Error Log"):
                        st.warning(result["stderr"])
            else:
                st.error(f"❌ Agent execution failed: {result.get('error', 'Unknown error')}")
                if result.get('traceback'):
                    with st.expander("View Traceback"):
                        st.code(result['traceback'], language='python')
        
        st.rerun()

def main():
    """Main entry point with tab navigation."""
    # Create tabs for different views
    if API_CLIENTS_AVAILABLE:
        tab1, tab2 = st.tabs(["Workflow Agents", "Task & Agent Management"])
        
        with tab1:
            create_main_ui()
        
        with tab2:
            create_task_management_ui()
    else:
        # If API clients not available, just show the workflow UI
        create_main_ui()

main()

