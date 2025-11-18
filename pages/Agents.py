import streamlit as st
import sys
import os
import json
import logging
from pathlib import Path

# Add parent directory to path to import alt_flows modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging to capture agent output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import agent workflows (with optional imports for dependencies)
try:
    from alt_flows.multi_agent import run_agentic_workflow
    MULTI_AGENT_AVAILABLE = True
except ImportError as e:
    MULTI_AGENT_AVAILABLE = False
    logger.warning(f"Multi-agent workflow not available: {e}")

try:
    from alt_flows.multi_agent_crewai import run_crewai_workflow
    CREWAI_AVAILABLE = True
except ImportError as e:
    CREWAI_AVAILABLE = False
    logger.warning(f"CrewAI workflow not available: {e}")

try:
    from alt_flows.main import agent_1, agent_2
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

def create_sidebar():
    """Create sidebar with agent selection and settings."""
    with st.sidebar:
        st.title("Agent Settings")
        
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

create_main_ui()

