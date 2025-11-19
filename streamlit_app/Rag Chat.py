import streamlit as st
import os
from lib.clients.chat_client import ChatClient

# --- Config ---
API_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8001")

st.set_page_config(page_title="RAG Chat", layout="wide")

# Initialize chat client
chat_client = ChatClient(base_url=API_URL)

def set_session_state():
    # --- Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "available_models" not in st.session_state:
        try:
            models = chat_client.get_models()
            st.session_state.available_models = models if models else ["ollama/llama3.1"]
        except Exception as e:
            st.session_state.available_models = ["ollama/llama3.1"]  # Fallback model
            st.error(f"Failed to fetch models: {e}")

    if "selected_model" not in st.session_state:
        if st.session_state.available_models and len(st.session_state.available_models) > 0:
            # Use first model, but skip connection error messages
            valid_models = [m for m in st.session_state.available_models if not m.startswith("Connection Error")]
            st.session_state.selected_model = valid_models[0] if valid_models else st.session_state.available_models[0]
        else:
            st.session_state.selected_model = "ollama/llama3.1"
    if "use_rag" not in st.session_state:
        st.session_state.use_rag = False

def create_sidebar_settings():
    """Create a well-organized settings panel in the sidebar"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # --- Model Configuration Section ---
        st.subheader("🤖 Model Configuration")
        
        # Filter out connection error messages from model list
        valid_models = [m for m in st.session_state.available_models if not m.startswith("Connection Error")]
        if not valid_models:
            valid_models = ["ollama/llama3.1"]
        
        # Find current model index or default to 0
        current_index = 0
        if st.session_state.selected_model in valid_models:
            current_index = valid_models.index(st.session_state.selected_model)
        
        selected_model = st.selectbox(
            "**Model**", 
            valid_models,
            index=current_index,
            key="sidebar_model",
            help="Select the language model to use for chat"
        )
        
        # Update session state with selected model
        st.session_state.selected_model = selected_model
        
        use_rag = st.toggle(
            "**Enable RAG**", 
            value=st.session_state.use_rag, 
            key="sidebar_rag",
            help="Enable Retrieval-Augmented Generation using your knowledge base"
        )
        
        # Update session state with RAG setting
        st.session_state.use_rag = use_rag
        
        # RAG Status Indicator
        if use_rag:
            st.success("✅ RAG is active")
        else:
            st.info("ℹ️ RAG is disabled")
        
        st.divider()
        
        # --- Knowledge Base Section ---
        with st.expander("📚 Knowledge Base", expanded=use_rag):
            # Fetch and display files in vector database
            if "files_list" not in st.session_state:
                try:
                    files = chat_client.get_files_in_vector_db()
                    st.session_state.files_list = files
                except Exception as e:
                    st.session_state.files_list = []
                    st.error(f"Failed to fetch files: {e}")
            
            # Ingest button (prominent)
            if st.button("📥 Ingest Knowledge Folder", key="sidebar_ingest", use_container_width=True):
                with st.spinner("Ingesting documents..."):
                    try:
                        response = chat_client.ingest_knowledge()
                        st.success(f"✅ {response.files_processed} file(s) processed")
                        # Refresh file list after ingestion
                        if "files_list" in st.session_state:
                            del st.session_state.files_list
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Connection failed: {e}")
            
            st.caption("Process .txt and .md files from the knowledge folder")
            
            # File selection section
            if st.session_state.files_list and isinstance(st.session_state.files_list, list) and len(st.session_state.files_list) > 0:
                st.write("**Available Files**")
                
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
                        "Select files to use:",
                        options=st.session_state.files_list,
                        key="selected_files",
                        label_visibility="collapsed"
                    )
                except Exception as e:
                    st.error(f"Error with file selection: {e}")
                    selected_files = []
                
                # Selection summary
                if selected_files:
                    if len(selected_files) == len(st.session_state.files_list):
                        st.caption(f"✅ All {len(selected_files)} file(s) selected")
                    else:
                        st.caption(f"📄 {len(selected_files)} of {len(st.session_state.files_list)} file(s) selected")
                else:
                    st.caption("No files selected")
            else:
                st.info("📁 No files found in knowledge base")
                st.caption("Add .txt or .md files to the 'knowledge' folder and click 'Ingest Knowledge Folder'")
        
        st.divider()
        
        # --- Chat Actions Section ---
        st.subheader("💬 Chat Actions")
        
        if st.button("🗑️ Clear Chat History", key="sidebar_clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.success("Chat cleared!")
            st.rerun()
        
        st.caption("Remove all messages from the current session")

def create_main_chat_ui():
    st.title("🦙 Llama Stack RAG Chat")

    # Display History - Show all messages from session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input - Process new messages
    if prompt := st.chat_input("What would you like to know?"):
        # Validate that we have a valid model
        if not st.session_state.selected_model or st.session_state.selected_model.startswith("Connection Error"):
            st.error("⚠️ Please select a valid model from the sidebar settings.")
            st.stop()
        
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Call Backend - show spinner while waiting
        bot_response = None
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chat_client.chat_message(
                        message=prompt,
                        model=st.session_state.selected_model,
                        session_id=st.session_state.session_id,
                        use_rag=st.session_state.use_rag
                    )
                    bot_response = response.response
                    # Update Session ID (in case it was created new)
                    st.session_state.session_id = response.session_id
                    
                    # Display the response immediately
                    st.markdown(bot_response)
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    bot_response = error_msg
                    st.error(error_msg)
                    # Show more details in expander for debugging
                    with st.expander("Error Details"):
                        st.exception(e)
        
        # Add assistant response to history after displaying
        if bot_response:
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            # Rerun to update the UI and show the message in the history
            st.rerun()

set_session_state()
create_sidebar_settings()
create_main_chat_ui()