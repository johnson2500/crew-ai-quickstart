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
API_URL = os.getenv("API_URL", "http://localhost:8001")

# No need to import workflow agents - this page only creates agents and tasks

st.set_page_config(page_title="Create Agent", layout="wide")

# --- Session State ---
# No session state needed for this page - all data comes from API

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

# launch_task_api removed - launching is done in Agents.py

def create_sidebar():
    """Create sidebar with configuration settings."""
    with st.sidebar:
        st.title("Configuration")

        try:
            agents = fetch_agents()
            tasks = fetch_tasks()
            if agents:
                st.metric("Agents Created", len(agents))
            if tasks:
                st.metric("Tasks Created", len(tasks))
        except:
            pass


def create_crewai_agent_ui():
    """Create UI for building CrewAI agents and tasks."""
    st.title("🏗️ Create Agents & Tasks")
    st.markdown("Build custom CrewAI agents and tasks for your workflows.")
    
    # Tabs for Agents and Tasks
    tab1, tab2 = st.tabs(["🤖 Agents", "📋 Tasks"])
    
    with tab1:
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
    
# Main page - only create agents and tasks
create_sidebar()
create_crewai_agent_ui()

