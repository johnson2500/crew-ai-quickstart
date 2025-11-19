import streamlit as st
import sys
import os
import json
import logging
import requests
from pathlib import Path

# Add parent directory to path to import alt_flows modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging to capture agent output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Configuration ---
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Import agent workflows (with optional imports for dependencies)
try:
    from API.alt_flows.multi_agent import run_agentic_workflow
    MULTI_AGENT_AVAILABLE = True
except ImportError as e:
    MULTI_AGENT_AVAILABLE = False
    logger.warning(f"Multi-agent workflow not available: {e}")

try:
    from API.alt_flows.multi_agent_crewai import run_crewai_workflow
    CREWAI_AVAILABLE = True
except ImportError as e:
    CREWAI_AVAILABLE = False
    logger.warning(f"CrewAI workflow not available: {e}")

try:
    from API.alt_flows.main import agent_1, agent_2
    LLAMA_STACK_AGENTS_AVAILABLE = True
except ImportError as e:
    LLAMA_STACK_AGENTS_AVAILABLE = False
    logger.warning(f"Llama Stack agents not available: {e}")

st.set_page_config(page_title="Create Agent", layout="wide")

# --- Session State ---
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "agent_results" not in st.session_state:
    st.session_state.agent_results = []
if "crewai_agents" not in st.session_state:
    st.session_state.crewai_agents = []
if "crewai_tasks" not in st.session_state:
    st.session_state.crewai_tasks = []

# Initialize page state - default to "Create Agents & Tasks" since this is the CreateAgent page
if "current_page" not in st.session_state:
    st.session_state.current_page = "Create Agents & Tasks"

# --- API Helper Functions ---
def fetch_agents():
    """Fetch agents from the API."""
    try:
        response = requests.get(f"{API_URL}/agents")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch agents: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching agents: {str(e)}")
        return []

def create_agent_api(agent_data):
    """Create an agent via the API."""
    try:
        response = requests.post(f"{API_URL}/agents", json=agent_data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to create agent: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        return None

def fetch_tasks():
    """Fetch tasks from the API."""
    try:
        response = requests.get(f"{API_URL}/tasks")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch tasks: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching tasks: {str(e)}")
        return []

def create_task_api(task_data):
    """Create a task via the API."""
    try:
        response = requests.post(f"{API_URL}/tasks", json=task_data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to create task: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error creating task: {str(e)}")
        return None

def launch_task_api(task_id, agent_id):
    """Launch a task via the API."""
    try:
        response = requests.post(
            f"{API_URL}/launch-task",
            json={"task_id": task_id, "agent_id": agent_id},
            timeout=300  # 5 minute timeout for long-running tasks
        )
        if response.status_code == 200:
            result = response.json()
            # Store in launch history
            if "launch_history" not in st.session_state:
                st.session_state.launch_history = []
            st.session_state.launch_history.append(result)
            return result
        else:
            error_msg = f"Failed to launch task: {response.status_code}"
            try:
                error_detail = response.json().get("detail", response.text)
                error_msg += f" - {error_detail}"
            except:
                error_msg += f" - {response.text}"
            return {"success": False, "error": error_msg, "status": "failed"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Task execution timed out (exceeded 5 minutes)", "status": "timeout"}
    except Exception as e:
        return {"success": False, "error": str(e), "status": "error"}

def create_sidebar():
    """Create sidebar with navigation and page-specific settings."""
    with st.sidebar:
        st.title("Navigation")
        
        # Page navigation
        page = st.radio(
            "Select Page",
            ["Agent Trigger", "Create Agents & Tasks", "Launch Tasks"],
            index=1 if st.session_state.current_page == "Create Agents & Tasks" else (2 if st.session_state.current_page == "Launch Tasks" else 0),
            key="page_navigation"
        )
        
        # Update page state
        if page != st.session_state.current_page:
            st.session_state.current_page = page
            st.rerun()
        
        st.divider()
        
        # Page-specific settings
        if st.session_state.current_page == "Agent Trigger":
            st.subheader("Agent Settings")
            
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
        else:
            # Settings for Create Agents & Tasks page
            st.subheader("Configuration")
            
            st.info("💡 **Tip:** Create agents and tasks to build custom CrewAI workflows.")
            
            if st.session_state.crewai_agents or st.session_state.crewai_tasks:
                st.metric("Agents Created", len(st.session_state.crewai_agents))
                st.metric("Tasks Created", len(st.session_state.crewai_tasks))
            
            return None, None

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

def create_crewai_agent_ui():
    """Create UI for building CrewAI agents and tasks."""
    st.title("🏗️ Create CrewAI Agents & Tasks")
    st.markdown("Build custom CrewAI agents and tasks for your workflows.")
    
    # Refresh button to fetch from API
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"💡 **API Endpoint:** {API_URL}")
    with col2:
        if st.button("🔄 Refresh from API"):
            st.rerun()
    
    # Tabs for Agents and Tasks
    tab1, tab2 = st.tabs(["🤖 Agents", "📋 Tasks"])
    
    with tab1:
        st.subheader("Create New Agent")

        st.info("💡 **Tip:** This form is based on creating agents for CrewAI workflows. See the CrewAI documentation for more information.")
        
        with st.form("create_agent_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                agent_role = st.text_input("Role *", placeholder="e.g., Weather Specialist")
                agent_goal = st.text_area("Goal *", placeholder="e.g., Get accurate weather information for any location")
                agent_backstory = st.text_area("Backstory *", placeholder="e.g., You are an expert meteorologist...")
            
            with col2:
                agent_model = st.selectbox(
                    "Model *",
                    ["ollama/llama3.2:3b", "meta-llama/Llama-3.1-8B-Instruct"],
                    index=0
                )
                agent_verbose = st.checkbox("Verbose", value=True)
                agent_allow_delegation = st.checkbox("Allow Delegation", value=False)
                
                # Tools selection (placeholder for now)
                st.markdown("**Tools**")
                st.info("💡 Tool creation UI coming soon. For now, agents can be created without tools.")
            
            submitted = st.form_submit_button("➕ Add Agent", type="primary")
            
            if submitted:
                if agent_role and agent_goal and agent_backstory:
                    new_agent = {
                        "role": agent_role,
                        "goal": agent_goal,
                        "backstory": agent_backstory,
                        "verbose": agent_verbose,
                        "allow_delegation": agent_allow_delegation,
                        "tools": []  # Placeholder for tools
                    }
                    result = create_agent_api(new_agent)
                    if result:
                        st.success(f"✅ Agent '{agent_role}' created successfully!")
                        st.rerun()
                else:
                    st.error("Please fill in all required fields (marked with *)")
        
        # Fetch and display existing agents from API
        agents = fetch_agents()
        if agents:
            st.divider()
            st.subheader(f"Created Agents ({len(agents)})")
            
            for agent in agents:
                with st.expander(f"🤖 {agent.get('role', 'Unknown')} (ID: {agent.get('id', 'N/A')})", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**ID:** {agent.get('id', 'N/A')}")
                        st.write(f"**Goal:** {agent.get('goal', 'N/A')}")
                        st.write(f"**Backstory:** {agent.get('backstory', 'N/A')}")
                        st.write(f"**Verbose:** {agent.get('verbose', False)}")
                        st.write(f"**Allow Delegation:** {agent.get('allow_delegation', False)}")
                        st.write(f"**Tools:** {len(agent.get('tools', []))} tool(s)")
                    
                    with col2:
                        st.write(f"**Agent ID:**")
                        st.code(agent.get('id', 'N/A'), language='text')
        else:
            st.info("No agents found. Create one above!")
    
    with tab2:
        st.subheader("Create New Task")
        
        # Fetch agents from API
        agents = fetch_agents()
        
        # Check if agents exist
        if not agents:
            st.warning("⚠️ Please create at least one agent before creating tasks.")
        else:
            with st.form("create_task_form", clear_on_submit=True):
                task_name = st.text_input("Task Name *", placeholder="e.g., Get Weather Information")
                
                task_description = st.text_area(
                    "Task Description *",
                    placeholder="e.g., Get the weather information for Boston, MA",
                    height=100
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Agent selection
                    agent_options = {agent['id']: f"{agent.get('role', 'Unknown')} (ID: {agent['id']})" for agent in agents}
                    selected_agent_id = st.selectbox(
                        "Assign to Agent *",
                        options=list(agent_options.keys()),
                        format_func=lambda x: agent_options[x]
                    )
                    
                    expected_output = st.text_area(
                        "Expected Output *",
                        placeholder="e.g., JSON string with temperature and conditions",
                        height=80
                    )
                
                with col2:
                    # Context/Dependencies (other tasks)
                    tasks = fetch_tasks()
                    if tasks:
                        st.markdown("**Task Dependencies (Context)**")
                        task_options = {task['id']: f"{task.get('name', 'Task')}: {task.get('description', '')[:40]}..." for task in tasks}
                        selected_deps = st.multiselect(
                            "Select dependencies",
                            options=list(task_options.keys()),
                            format_func=lambda x: task_options[x],
                            help="Select tasks that this task depends on (context)"
                        )
                        task_dependencies = list(selected_deps)
                    else:
                        st.info("No existing tasks to depend on.")
                        task_dependencies = []
                
                submitted = st.form_submit_button("➕ Add Task", type="primary")
                
                if submitted:
                    if task_name and task_description and expected_output:
                        new_task = {
                            "name": task_name,
                            "description": task_description,
                            "agent_id": selected_agent_id,
                            "expected_output": expected_output,
                            "dependencies": task_dependencies
                        }
                        result = create_task_api(new_task)
                        if result:
                            selected_agent = next((a for a in agents if a['id'] == selected_agent_id), None)
                            agent_role = selected_agent.get('role', 'Unknown') if selected_agent else 'Unknown'
                            st.success(f"✅ Task created and assigned to '{agent_role}'!")
                            st.rerun()
                    else:
                        st.error("Please fill in all required fields (marked with *)")
            
            # Fetch and display existing tasks from API
            tasks = fetch_tasks()
            if tasks:
                st.divider()
                st.subheader(f"Created Tasks ({len(tasks)})")
                
                for task in tasks:
                    task_name = task.get('name', 'Unnamed Task')
                    task_desc = task.get('description', 'No description')[:60]
                    with st.expander(f"📋 {task_name}: {task_desc}... (ID: {task.get('id', 'N/A')})", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**ID:** {task.get('id', 'N/A')}")
                            st.write(f"**Name:** {task.get('name', 'N/A')}")
                            st.write(f"**Description:** {task.get('description', 'N/A')}")
                            st.write(f"**Agent ID:** {task.get('agent_id', 'N/A')}")
                            st.write(f"**Expected Output:** {task.get('expected_output', 'N/A')}")
                            if task.get('dependencies'):
                                st.write(f"**Dependencies:** {', '.join(task['dependencies'])}")
                            else:
                                st.write("**Dependencies:** None")
                        
                        with col2:
                            st.write(f"**Task ID:**")
                            st.code(task.get('id', 'N/A'), language='text')
            else:
                st.info("No tasks found. Create one above!")
    
    # Export/Import section
    st.divider()
    st.subheader("💾 Export/Import Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Configuration"):
            config = {
                "agents": st.session_state.crewai_agents,
                "tasks": st.session_state.crewai_tasks
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(config, indent=2),
                file_name="crewai_config.json",
                mime="application/json"
            )
    
    with col2:
        uploaded_file = st.file_uploader("📤 Import Configuration", type=["json"])
        if uploaded_file is not None:
            try:
                config = json.load(uploaded_file)
                if "agents" in config and "tasks" in config:
                    st.session_state.crewai_agents = config["agents"]
                    st.session_state.crewai_tasks = config["tasks"]
                    st.success("✅ Configuration imported successfully!")
                    st.rerun()
                else:
                    st.error("Invalid configuration file format.")
            except json.JSONDecodeError:
                st.error("Invalid JSON file.")
    
    # Clear all button
    if st.session_state.crewai_agents or st.session_state.crewai_tasks:
        st.divider()
        if st.button("🗑️ Clear All Agents & Tasks", type="secondary"):
            st.session_state.crewai_agents = []
            st.session_state.crewai_tasks = []
            st.success("✅ All agents and tasks cleared!")
            st.rerun()

def create_main_ui(agent_type, model):
    """Create the main agent triggering UI."""
    st.title("🤖 Agent Trigger")
    st.markdown("Trigger and interact with various agent workflows.")
    
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

def create_launch_task_ui():
    """Create UI for launching tasks via the API."""
    st.title("🚀 Launch Tasks")
    st.markdown("Launch CrewAI tasks using agents and tasks created via the API.")
    
    # Refresh button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"💡 **API Endpoint:** {API_URL}")
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Fetch agents and tasks from API
    agents = fetch_agents()
    tasks = fetch_tasks()
    
    if not agents:
        st.warning("⚠️ No agents found. Please create at least one agent first.")
        st.info("Go to 'Create Agents & Tasks' page to create agents.")
        return
    
    if not tasks:
        st.warning("⚠️ No tasks found. Please create at least one task first.")
        st.info("Go to 'Create Agents & Tasks' page to create tasks.")
        return
    
    st.divider()
    st.subheader("Launch Task")
    
    with st.form("launch_task_form"):
        # Task selection
        task_options = {task['id']: f"{task.get('name', 'Unnamed')}: {task.get('description', '')[:50]}..." for task in tasks}
        selected_task_id = st.selectbox(
            "Select Task *",
            options=list(task_options.keys()),
            format_func=lambda x: task_options[x],
            help="Select the task you want to launch"
        )
        
        # Agent selection
        agent_options = {agent['id']: f"{agent.get('role', 'Unknown')} (ID: {agent['id']})" for agent in agents}
        selected_agent_id = st.selectbox(
            "Select Agent *",
            options=list(agent_options.keys()),
            format_func=lambda x: agent_options[x],
            help="Select the agent to execute this task"
        )
        
        # Display task and agent details
        if selected_task_id:
            selected_task = next((t for t in tasks if t['id'] == selected_task_id), None)
            if selected_task:
                with st.expander("📋 Task Details"):
                    st.write(f"**Name:** {selected_task.get('name', 'N/A')}")
                    st.write(f"**Description:** {selected_task.get('description', 'N/A')}")
                    st.write(f"**Expected Output:** {selected_task.get('expected_output', 'N/A')}")
                    if selected_task.get('dependencies'):
                        st.write(f"**Dependencies:** {', '.join(selected_task['dependencies'])}")
        
        if selected_agent_id:
            selected_agent = next((a for a in agents if a['id'] == selected_agent_id), None)
            if selected_agent:
                with st.expander("🤖 Agent Details"):
                    st.write(f"**Role:** {selected_agent.get('role', 'N/A')}")
                    st.write(f"**Goal:** {selected_agent.get('goal', 'N/A')}")
                    st.write(f"**Backstory:** {selected_agent.get('backstory', 'N/A')}")
        
        submitted = st.form_submit_button("🚀 Launch Task", type="primary")
        
        if submitted:
            if selected_task_id and selected_agent_id:
                with st.spinner("Launching task... This may take a while."):
                    result = launch_task_api(selected_task_id, selected_agent_id)
                    
                    if result and result.get("status") == "completed":
                        st.success("✅ Task launched successfully!")
                        
                        st.subheader("Execution Result")
                        st.write(f"**Task ID:** {result.get('task_id', 'N/A')}")
                        st.write(f"**Agent ID:** {result.get('agent_id', 'N/A')}")
                        st.write(f"**Status:** {result.get('status', 'N/A')}")
                        
                        if result.get('result'):
                            st.write("**Result:**")
                            st.code(result['result'], language='text')
                    else:
                        st.error(f"❌ Task launch failed: {result.get('error', 'Unknown error') if result else 'No response from API'}")
            else:
                st.error("Please select both a task and an agent.")
    
    # Display launch history
    if "launch_history" not in st.session_state:
        st.session_state.launch_history = []
    
    if st.session_state.launch_history:
        st.divider()
        st.subheader("Launch History")
        for i, launch in enumerate(reversed(st.session_state.launch_history[-10:])):  # Show last 10
            with st.expander(f"Launch #{len(st.session_state.launch_history) - i}: {launch.get('task_id', 'N/A')} - {launch.get('status', 'N/A')}"):
                st.json(launch)

# Main page routing based on sidebar selection
agent_type, model = create_sidebar()

if st.session_state.current_page == "Create Agents & Tasks":
    create_crewai_agent_ui()
elif st.session_state.current_page == "Launch Tasks":
    create_launch_task_ui()
else:
    # Check if sidebar returned None (no agents available)
    if agent_type is None or model is None:
        st.stop()
    create_main_ui(agent_type, model)

