import streamlit as st
import requests
import uuid

# --- Config ---
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Llama Stack Chat", layout="wide")


def set_session_state():
    # --- Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "available_models" not in st.session_state:
        try:
            res = requests.get(f"{API_URL}/models")
            if res.status_code == 200:
                st.session_state.available_models = res.json()
            else:
                st.session_state.available_models = ["meta-llama/Llama-3.1-8B-Instruct"]
        except:
            st.session_state.available_models = ["Connection Error"]

# --- Sidebar ---
with st.sidebar:
    st.title("Settings")
    
    selected_model = st.selectbox(
        "Select Model", 
        st.session_state.available_models,
        index=0
    )
    
    use_rag = st.toggle("Enable RAG (Knowledge Base)", value=False)
    
    st.divider()
    st.subheader("Knowledge Base")
    if st.button("Ingest Knowledge Folder"):
        with st.spinner("Ingesting documents..."):
            try:
                res = requests.post(f"{API_URL}/ingest")
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Success! {data['files_processed']} files processed.")
                    # Refresh file list after ingestion
                    if "files_list" in st.session_state:
                        del st.session_state.files_list
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")
    
    # Fetch and display files in vector database
    if "files_list" not in st.session_state:
        try:
            res = requests.get(f"{API_URL}/files-in-vector-db")
            if res.status_code == 200:
                st.session_state.files_list = res.json()
            else:
                st.session_state.files_list = []
        except Exception as e:
            st.session_state.files_list = []
            st.error(f"Failed to fetch files: {e}")
    
    # Display file selection multiselect dropdown
    if st.session_state.files_list and isinstance(st.session_state.files_list, list) and len(st.session_state.files_list) > 0:
        st.subheader("Files in Vector DB")
        
        # Initialize selected_files in session state if not present
        if "selected_files" not in st.session_state:
            st.session_state.selected_files = []
        
        # Ensure selected_files is a list
        if not isinstance(st.session_state.selected_files, list):
            st.session_state.selected_files = []
        
        # Determine if all files are currently selected
        current_selection = st.session_state.selected_files
        all_selected = len(current_selection) == len(st.session_state.files_list) and len(st.session_state.files_list) > 0
        
        # Select All checkbox - sync with current selection
        select_all = st.checkbox("Select All", value=all_selected, key="select_all_checkbox")
        
        # Update selection based on checkbox state
        if select_all:
            # If checkbox is checked, ensure all files are selected
            if len(current_selection) != len(st.session_state.files_list):
                st.session_state.selected_files = st.session_state.files_list.copy()
        else:
            # If checkbox is unchecked and all were selected, clear selection
            if all_selected:
                st.session_state.selected_files = []
        
        # Use multiselect with key
        try:
            selected_files = st.multiselect(
                "Select files (can select multiple):",
                options=st.session_state.files_list,
                key="selected_files"
            )
        except Exception as e:
            st.error(f"Error with file selection: {e}")
            selected_files = []
        
        if selected_files:
            if len(selected_files) == len(st.session_state.files_list):
                st.info(f"✅ All {len(selected_files)} file(s) selected: **{', '.join(selected_files)}**")
            else:
                st.info(f"Selected {len(selected_files)} of {len(st.session_state.files_list)} file(s): **{', '.join(selected_files)}**")
        else:
            st.info("No files selected.")
    else:
        st.info("No files found in vector database.")
    
    st.info("Place .txt or .md files in the 'knowledge' folder in the root directory.")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

# --- Main Chat UI ---
st.title("🦙 Llama Stack RAG Chat")

# Display History - Show all messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input - Process new messages
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Prepare Payload
    payload = {
        "message": prompt,
        "model": selected_model,
        "session_id": st.session_state.session_id,
        "use_rag": use_rag
    }

    # Call Backend - show spinner while waiting
    bot_response = None
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(f"{API_URL}/chat", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    bot_response = data["response"]
                    # Update Session ID (in case it was created new)
                    st.session_state.session_id = data["session_id"]
                    
                    # Display the response immediately
                    st.markdown(bot_response)
                else:
                    error_msg = f"Backend Error: {response.text}"
                    bot_response = error_msg
                    st.error(error_msg)
            except Exception as e:
                error_msg = f"Connection Error: {e}"
                bot_response = error_msg
                st.error(error_msg)
    
    # Add assistant response to history after displaying
    if bot_response:
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        # Rerun to update the UI and show the message in the history
        st.rerun()