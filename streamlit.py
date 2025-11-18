import streamlit as st
import requests
import uuid

# --- Config ---
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Llama Stack Chat", layout="wide")

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
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")
    
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