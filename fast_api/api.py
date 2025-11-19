import os
import glob
import uuid
import json
import requests
import traceback
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_stack_client import LlamaStackClient
from lib.models import ChatRequest, ChatResponse, IngestResponse
from lib.logger import logger
from lib.prompts import RAG_SYSTEM_PROMPT, RAG_NO_CONTEXT_PROMPT, NORMAL_SYSTEM_PROMPT
from config import LLAMA_STACK_URL, KNOWLEDGE_FOLDER, VECTOR_DB_ID, SESSION_STORAGE

# Default Ollama model for all agents
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama/llama3.2:3b")
from lib.llama_stack_client import LlamaClient
from crewai import Crew, Task, Agent
from lib.llama_stack_llm import LlamaStackLLM   
# Ensure knowledge folder exists
os.makedirs(KNOWLEDGE_FOLDER, exist_ok=True)

app = FastAPI(title="Llama Stack RAG Backend")
agents = []
tasks = []
try:
    client = LlamaStackClient(base_url=LLAMA_STACK_URL)
    logger.info(f"DEBUG: Client Attributes: {[x for x in dir(client) if not x.startswith('_')]}")
except Exception as e:
    logger.info(f"SDK Connection Error: {e}")
    client = None

def raw_chat_completion(model_id: str, messages: List[Dict]):
    """Direct HTTP call to Llama Stack Inference API."""
    url = f"{LLAMA_STACK_URL}/v1/chat/completions"
    payload = {"model": model_id, "messages": messages, "stream": False}
    logger.info(f"DEBUG: Sending Raw Request to {url} with model {model_id}")
    
    resp = requests.post(f"{LLAMA_STACK_URL}/v1/chat/completions", json=payload)
        
    resp.raise_for_status()
    return resp.json()

def raw_vector_query(query: str):
    """Direct HTTP call to Llama Stack Vector IO Query."""
    url = f"{LLAMA_STACK_URL}/v1/vector_io/query"
    payload = {
        "vector_db_id": VECTOR_DB_ID,
        "query": query,
        "params": {"top_k": 3}
    }
    try:
        resp = requests.post(url, json=payload)

        logger.info(f"Raw Vector Query Response: {resp.json()}")
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.info(f"Raw Vector Query Error: {e}")
    return None

def raw_ingest_documents(documents: List[Dict]):
    """Direct HTTP call to insert documents."""
    # Try standard vector_io insert
    url = f"{LLAMA_STACK_URL}/v1/tool-runtime/rag-tool/insert"
    payload = {
        "chunk_size_in_tokens": 512,
        "documents": documents,
        "vector_store_id": VECTOR_DB_ID
    }

    logger.info(f"Raw Ingest Documents Payload: {payload}")
    
    resp = requests.post(url, json=payload)

    if resp.status_code == 200:
        logger.info(f"Raw Ingest Documents Response: {resp.json()}")
        return resp.json()
    else:
        logger.info(f"Raw Ingest Documents Error: {resp.status_code} {resp.text}")
        return None


# --- Helper Functions: Logic ---

def get_or_create_vector_db():
    """Creates a vector database if it doesn't exist, following the pattern from upload.py."""
    # First, try to check if vector database exists via SDK
    vector_db_exists = False
    
    # Try SDK approach first (if vector_dbs is available)
    if client:
        try:
            # Check if vector_dbs attribute exists (may not be in all client versions)
            if hasattr(client, 'vector_dbs'):
                stores = client.vector_dbs.list()
                for store in stores:
                    if hasattr(store, 'identifier') and store.identifier == VECTOR_DB_ID:
                        vector_db_exists = True
                        logger.info(f"Vector DB '{VECTOR_DB_ID}' already exists")
                        return VECTOR_DB_ID
                    elif hasattr(store, 'id') and store.id == VECTOR_DB_ID:
                        vector_db_exists = True
                        logger.info(f"Vector DB '{VECTOR_DB_ID}' already exists")
                        return VECTOR_DB_ID
        except Exception as e:
            logger.info(f"Could not check vector_dbs via SDK: {e}")
        
        # Try vector_stores as alternative
        if not vector_db_exists and hasattr(client, 'vector_stores'):
            try:
                stores = client.vector_stores.list()
                for store in stores:
                    if hasattr(store, 'id') and store.id == VECTOR_DB_ID:
                        vector_db_exists = True
                        logger.info(f"Vector Store '{VECTOR_DB_ID}' already exists")
                        return VECTOR_DB_ID
            except Exception as e:
                logger.info(f"Could not check vector_stores: {e}")
    
    # If not found, create it
    if not vector_db_exists:
        logger.info(f"Vector DB '{VECTOR_DB_ID}' not found, creating new one...")
        
        # Get the vector_io provider_id
        provider_id = None
        if client and hasattr(client, 'providers'):
            try:
                providers = client.providers.list()
                for provider in providers:
                    if hasattr(provider, 'api') and provider.api == "vector_io":
                        provider_id = provider.provider_id
                        break
            except Exception as e:
                logger.info(f"Could not get providers: {e}")
        
        # Default provider if not found
        if not provider_id:
            provider_id = "faiss"  # Default to faiss for local storage
            logger.info(f"Using default provider: {provider_id}")
        
        # Try to register via SDK
        if client and hasattr(client, 'vector_dbs'):
            try:
                vector_db = client.vector_dbs.register(
                    vector_db_id=VECTOR_DB_ID,
                    embedding_dimension=384,
                    embedding_model="all-MiniLM-L6-v2",
                    provider_id=provider_id
                )
                logger.info(f"Successfully created Vector DB '{VECTOR_DB_ID}' via SDK")
                return VECTOR_DB_ID
            except Exception as e:
                logger.info(f"SDK vector_dbs.register failed: {e}")
        
        # Fallback to REST API
        try:
            url = f"{LLAMA_STACK_URL}/v1/vector_dbs/register"
            payload = {
                "vector_db_id": VECTOR_DB_ID,
                "embedding_dimension": 384,
                "embedding_model": "all-MiniLM-L6-v2",
                "provider_id": provider_id
            }
            resp = requests.post(url, json=payload)
            if resp.status_code == 200:
                logger.info(f"Successfully created Vector DB '{VECTOR_DB_ID}' via REST API")
                return VECTOR_DB_ID
            elif resp.status_code == 409:  # Already exists
                logger.info(f"Vector DB '{VECTOR_DB_ID}' already exists (409)")
                return VECTOR_DB_ID
            else:
                logger.info(f"Failed to create vector DB via REST: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.info(f"REST API vector_dbs.register failed: {e}")
            logger.info(f"Note: Vector DB may be created automatically on first insert")
    
    return VECTOR_DB_ID

def perform_manual_rag(query: str) -> str:
    """
    return the context string for the query
    """
    logger.info(f"🔍 RAG PIPELINE TRIGGERED - Query: '{query[:100]}...' (truncated)" if len(query) > 100 else f"🔍 RAG PIPELINE TRIGGERED - Query: '{query}'")
    logger.info(f"📊 RAG Configuration - Vector DB ID: {VECTOR_DB_ID} - Client: {client}")
    context_parts = []
    
    # 1. Try SDK rag_tool.query (Preferred method - matches tool_runtime.py API)
    if client and hasattr(client, 'tool_runtime') and hasattr(client.tool_runtime, 'rag_tool'):
        logger.info("🔄 Attempting RAG query via rag_tool.query (REST API)")
        logger.info(f"🔑 Using Vector Store ID: {VECTOR_DB_ID}")
        try:
            logger.info("Sending RAG Query to REST API")
            results = requests.post(
                f"{LLAMA_STACK_URL}/v1/tool-runtime/rag-tool/query",
                json={"content": query, "vector_store_ids": [VECTOR_DB_ID]},
                timeout=30
            )
            data = results.json()
            logger.info(f"Raw RAG Query Response: {data}")
            
            # Parse the response - rag_tool.query may return content directly or in chunks
            if not isinstance(data, dict):
                logger.warning("RAG query returned non-dict response")
            elif "chunks" in data and data["chunks"]:
                # Extract content from chunks
                items = data["chunks"] if isinstance(data["chunks"], list) else [data["chunks"]]
                for item in items:
                    if isinstance(item, dict):
                        text = item.get("content") or item.get("text", "")
                    else:
                        text = str(item)
                    if text:
                        context_parts.append(text)
            elif "content" in data:
                # Extract direct content field
                if data["content"]:
                    context_parts.append(data["content"])
                else:
                    logger.warning("RAG query returned empty content")
            else:
                # Fallback: try other common response keys
                for key in ["text", "message", "result"]:
                    if key in data and data[key]:
                        context_parts.append(str(data[key]))
                        break
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            import traceback
            traceback.logger.info_exc()
    
    if not context_parts:
        logger.warning("⚠️  RAG PIPELINE COMPLETED - No context retrieved from knowledge base")
        return ""

    # Filter and convert context_parts to strings, removing None/empty values
    context_parts_clean = []
    for part in context_parts:
        if part is not None:
            part_str = str(part).strip()
            if part_str:
                context_parts_clean.append(part_str)
    
    if not context_parts_clean:
        logger.warning("⚠️  RAG PIPELINE COMPLETED - No valid context retrieved from knowledge base")
        return ""

    total_context_length = sum(len(cp) for cp in context_parts_clean)
    logger.info(f"✅ RAG PIPELINE COMPLETED - Retrieved {len(context_parts_clean)} chunks, {total_context_length} total characters")
    context_str = "\n\n---\nRelevant Context from Knowledge Base:\n" + "\n".join(context_parts_clean) + "\n---\n"
    logger.debug(f"📝 Final context string length: {len(context_str)} characters")
    return context_str

# --- Endpoints ---

@app.get("/health")
def health_check():
    """Health check endpoint to verify API and Llama Stack connection."""
    status = {
        "api": "healthy",
        "llama_stack_url": LLAMA_STACK_URL,
        "client_connected": client is not None,
        "vector_db_id": VECTOR_DB_ID,
        "knowledge_folder": KNOWLEDGE_FOLDER,
        "knowledge_folder_exists": os.path.exists(KNOWLEDGE_FOLDER)
    }
    
    # Check if knowledge folder has files
    if os.path.exists(KNOWLEDGE_FOLDER):
        files = glob.glob(os.path.join(KNOWLEDGE_FOLDER, "*.txt")) + \
                glob.glob(os.path.join(KNOWLEDGE_FOLDER, "*.md"))
        status["knowledge_files_count"] = len(files)
    
    return status

@app.get("/models")
def list_models():
    """Lists models but FILTERS OUT embedding models to prevent 400 errors."""
    models = []
    
    # 1. Try SDK
    if client and hasattr(client, 'models'):
        try:
            models = [m.identifier for m in client.models.list()]
        except: pass
    
    # 2. Try REST
    if not models:
        try:
            resp = requests.get(f"{LLAMA_STACK_URL}/models/list")
            if resp.status_code == 200:
                data = resp.json()
                # Handle different list shapes
                if isinstance(data, list): models = [m.get("identifier") for m in data]
                elif "data" in data: models = [m.get("identifier") for m in data["data"]]
        except: pass

    # 3. Fallback
    if not models:
        return ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]

    # --- CRITICAL FIX: Filter out non-chat models ---
    # 'all-minilm' caused your error because it cannot chat.
    chat_models = [
        m for m in models 
        if "minilm" not in m.lower() 
        and "bert" not in m.lower() 
        and "embedding" not in m.lower()
    ]
    
    # If filter killed everything (unlikely), return original list
    return chat_models if chat_models else models

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    # 1. Manage Session
    session_id = request.session_id or str(uuid.uuid4())

    # 2. Prepare RAG Context (do this first to determine system prompt)
    context_str = ""
    if request.use_rag:
        logger.info(f"🚀 RAG REQUESTED for session {session_id} - Model: {request.model}")
        logger.info(f"📝 User message: '{request.message[:100]}...' (truncated)" if len(request.message) > 100 else f"📝 User message: '{request.message}'")
        context_str = perform_manual_rag(request.message)
        if context_str:
            logger.info(f"✅ RAG context retrieved and will be added to prompt")
        else:
            logger.warning(f"⚠️  RAG was requested but no context was retrieved - proceeding without RAG context")
    else:
        logger.debug(f"ℹ️  RAG not requested (use_rag=False) for session {session_id}")
    
    # Determine system prompt based on RAG request and context availability
    if request.use_rag:
        system_prompt = RAG_SYSTEM_PROMPT if context_str else RAG_NO_CONTEXT_PROMPT
        if context_str:
            logger.info("🔒 RAG-only mode enabled: Model restricted to knowledge base context only")
        else:
            logger.info("🔒 RAG mode enabled but no context found")
    else:
        system_prompt = NORMAL_SYSTEM_PROMPT
    
    # Initialize or update session with appropriate system prompt
    if session_id not in SESSION_STORAGE:
        SESSION_STORAGE[session_id] = [{"role": "system", "content": system_prompt}]
    else:
        # Update existing session's system prompt
        if SESSION_STORAGE[session_id] and SESSION_STORAGE[session_id][0].get("role") == "system":
            SESSION_STORAGE[session_id][0]["content"] = system_prompt
            logger.info("🔒 Updated existing session system prompt")
        else:
            SESSION_STORAGE[session_id].insert(0, {"role": "system", "content": system_prompt})
            logger.info("🔒 Added system prompt to existing session")
    
    # Get history - use reference directly since we update SESSION_STORAGE in place above
    history = SESSION_STORAGE[session_id]
    
    # 3. Construct Prompt
    final_user_content = request.message
    if context_str and context_str.strip():
        # When RAG is used, format the context prominently with strict instructions
        # Extract just the context content (remove the wrapper)
        if 'Relevant Context from Knowledge Base:' in context_str:
            try:
                context_content = context_str.split('Relevant Context from Knowledge Base:')[1].split('---')[0].strip()
            except (IndexError, AttributeError):
                logger.warning("⚠️  Failed to parse context string, using full context_str")
                context_content = context_str.strip()
        else:
            context_content = context_str.strip()
        
        if context_content:
            final_user_content = f"""Relevant Context from Knowledge Base:
                {context_content}
                IMPORTANT: Answer the following question using ONLY the information from the 
                context above. Do not use any other knowledge. If the answer is not in the context, 
                clearly state that you don't have that information in your knowledge base.
                Question: {request.message}
            """
            logger.info(f"📋 Final prompt constructed with RAG context ({len(final_user_content)} characters total, context: {len(context_content)} chars)")
        else:
            logger.warning("⚠️  RAG context was retrieved but is empty after parsing - proceeding without context")
            logger.debug(f"📋 Final prompt constructed without RAG context ({len(final_user_content)} characters)")
    else:
        logger.debug(f"📋 Final prompt constructed without RAG context ({len(final_user_content)} characters)")

    # Prepare payload - ensure system prompt is included
    messages_payload = history + [{"role": "user", "content": final_user_content}]
    
    # Verify system prompt is present (should be first message)
    if messages_payload and messages_payload[0].get("role") != "system":
        logger.warning("⚠️  System prompt missing from messages_payload, adding default")
        messages_payload.insert(0, {"role": "system", "content": "You are a helpful assistant."})
    
    # Log the payload structure for debugging
    logger.debug(f"📤 Messages payload: {len(messages_payload)} messages (system: {messages_payload[0].get('role') == 'system'}, user: {len([m for m in messages_payload if m.get('role') == 'user'])} messages)")

    try:
        bot_content = ""
        
        # Strategy: Try SDK Inference -> Fail to REST Inference
        if client and hasattr(client, 'inference'):
            try:
                response = client.inference.chat_completion(
                    model_id=request.model,
                    messages=messages_payload,
                    stream=False
                )
                if hasattr(response, 'completion_message'):
                    bot_content = response.completion_message.content
                elif hasattr(response, 'choices'):
                    bot_content = response.choices[0].message.content
                else:
                    bot_content = str(response)
            except Exception as sdk_e:
                logger.error(f"SDK Inference failed ({sdk_e}), switching to REST...")
                raise Exception("Trigger REST Fallback") # Force catch block
        else:
            raise Exception("Inference API missing in SDK")

    except Exception:
        # Fallback to Raw REST
        try:
            data = raw_chat_completion(request.model, messages_payload)
            
            # Extract content from REST JSON
            if "completion_message" in data:
                bot_content = data["completion_message"].get("content", "")
            elif "choices" in data and len(data["choices"]) > 0:
                bot_content = data["choices"][0].get("message", {}).get("content", "")
            else:
                bot_content = str(data)
                
        except Exception as rest_e:
            logger.error(f"REST Inference failed ({rest_e}), switching to REST...") 
            raise HTTPException(status_code=500, detail=f"Llama Stack Error: {str(rest_e)}")

    # Clean up list content (multimodal)
    if isinstance(bot_content, list):
         bot_content = " ".join([
             getattr(b, 'text', '') or (b.get('text') if isinstance(b, dict) else str(b)) 
             for b in bot_content
         ])
    
    bot_content = str(bot_content)

    # 4. Update History
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": bot_content})
    logger.info(f"History: {history}")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Bot Content: {bot_content}")
    SESSION_STORAGE[session_id] = history

    return ChatResponse(response=bot_content, session_id=session_id)

@app.post("/ingest", response_model=IngestResponse)
def ingest_knowledge():
    files = glob.glob(os.path.join(KNOWLEDGE_FOLDER, "*.txt")) + \
            glob.glob(os.path.join(KNOWLEDGE_FOLDER, "*.md"))
    
    if not files:
        return IngestResponse(status="skipped", files_processed=0, details="No files found.")

    documents = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                file_name = os.path.basename(file_path)
                logger.info(f"📄 Found file: {file_name} ({len(content)} characters)")
                logger.debug(f"   Content preview: {content[:100]}...")
                documents.append({
                    "document_id": file_name,
                    "content": content,
                    "mime_type": "text/plain",
                    "metadata": {"filename": file_name}
                })
        except Exception as e:
            logger.error(f"❌ Failed to read file {file_path}: {e}")
    
    logger.info(f"📦 Prepared {len(documents)} documents for ingestion")

    try:
        get_or_create_vector_db()
        
        # Strategy: Try SDK Tool -> SDK VectorIO -> REST
        ingested = False

        if not ingested:
            logger.info("Using REST API to ingest documents")
            raw_ingest_documents(documents)

        logger.info(f"✅ Ingestion completed - {len(documents)} files processed")
        return IngestResponse(status="success", files_processed=len(documents), details=f"Inserted {len(documents)} files into vector database.")
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        traceback.logger.info_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.get("/files-in-vector-db", response_model=List[str])
def files_in_vector_db():
    """
    Get the files in the vector database (knowledge folder).
    Returns only .txt and .md files.
    """
    if not os.path.exists(KNOWLEDGE_FOLDER):
        return []
    
    files_and_directories = os.listdir(KNOWLEDGE_FOLDER)
    # Filter for only .txt and .md files
    files = [
        f for f in files_and_directories 
        if os.path.isfile(os.path.join(KNOWLEDGE_FOLDER, f)) 
        and f.endswith(('.txt', '.md'))
    ]
    
    return sorted(files)


class AgentBase(BaseModel):
    id: Optional[str] = None
    role: str
    goal: str
    backstory: str
    tools: List[str] = []
    verbose: bool = False
    allow_delegation: bool = False


def validate_agent(agent: AgentBase):
    if not agent.role:
        raise HTTPException(status_code=400, detail="Role is required")
    
    if not agent.goal:
        raise HTTPException(status_code=400, detail="Goal is required")
    
    if not agent.backstory:
        raise HTTPException(status_code=400, detail="Backstory is required")

    if agent.tools is None:
        agent.tools = []

    if agent.verbose is None:
        agent.verbose = False

    if agent.allow_delegation is None:
        agent.allow_delegation = False

    return agent

@app.get("/agents", response_model=List[AgentBase])
def get_agents():
    return agents

@app.post("/agents", response_model=AgentBase)
def create_agent(agent: AgentBase):
    try:    
        agent = validate_agent(agent)
        if agent.id is None:
            agent.id = str(uuid.uuid4())
        agents.append(agent)
        return agent
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/agents/{agent_id}", response_model=AgentBase)
def update_agent(agent_id: str, agent_update: AgentBase):
    for i, existing_agent in enumerate(agents):
        if existing_agent.id == agent_id:
            agent_update.id = agent_id
            agent_update = validate_agent(agent_update)
            agents[i] = agent_update
            return agent_update
    raise HTTPException(status_code=404, detail="Agent not found")

@app.delete("/agents/{agent_id}", response_model=AgentBase)
def delete_agent(agent_id: str):
    for i, agent in enumerate(agents):
        if agent.id == agent_id:
            deleted_agent = agents.pop(i)
            return deleted_agent
    raise HTTPException(status_code=404, detail="Agent not found")


@app.get("/agents/{agent_id}", response_model=AgentBase)
def get_agent(agent_id: str):
    for agent in agents:
        if agent.id == agent_id:
            return agent
    raise HTTPException(status_code=404, detail="Agent not found")


class TaskBase(BaseModel):
    id: Optional[str] = None
    name: str
    description: str
    agent_id: str
    expected_output: str
    dependencies: List[str] = []

# --- Initialize Base Agents and Tasks for Testing ---
def initialize_base_data():
    """Initialize the API with some base agents and tasks for easy testing."""
    global agents, tasks
    
    # Base Agents
    base_agents = [
        AgentBase(
            id=str(uuid.uuid4()),
            role="Weather Specialist",
            goal="Get accurate weather information for any location requested by the user",
            backstory="You are an expert meteorologist who always provides accurate, real-time weather data. Never guess or provide general information.",
            tools=[],
            verbose=True,
            allow_delegation=False
        ),
        AgentBase(
            id=str(uuid.uuid4()),
            role="Activity Planner",
            goal="Suggest appropriate outdoor activities based on current weather conditions",
            backstory="You are a local activity expert who knows the best things to do in any city based on weather. You provide thoughtful activity suggestions.",
            tools=[],
            verbose=True,
            allow_delegation=False
        ),
        AgentBase(
            id=str(uuid.uuid4()),
            role="Research Analyst",
            goal="Research and analyze information on any given topic",
            backstory="You are a thorough research analyst who gathers comprehensive information and provides detailed analysis on any topic.",
            tools=[],
            verbose=True,
            allow_delegation=True
        ),
        AgentBase(
            id=str(uuid.uuid4()),
            role="Content Writer",
            goal="Create well-written content based on research and requirements",
            backstory="You are a professional content writer who creates engaging, well-structured content based on provided information and requirements.",
            tools=[],
            verbose=True,
            allow_delegation=False
        )
    ]
    
    agents.extend(base_agents)
    
    # Base Tasks (created after agents are initialized)
    if agents:
        weather_agent_id = agents[0].id
        activity_agent_id = agents[1].id
        research_agent_id = agents[2].id
        writer_agent_id = agents[3].id
        
        base_tasks = [
            TaskBase(
                id=str(uuid.uuid4()),
                name="Get Weather Information",
                description="Get the current weather information for a specified location",
                agent_id=weather_agent_id,
                expected_output="JSON string with temperature, conditions, and location",
                dependencies=[],
            ),
            TaskBase(
                id=str(uuid.uuid4()),
                name="Suggest Activity",
                description="Based on weather conditions, suggest an appropriate outdoor activity",
                agent_id=activity_agent_id,
                expected_output="A description of a suggested outdoor activity with reasoning",
                dependencies=[]
            ),
            TaskBase(
                id=str(uuid.uuid4()),
                name="Research Topic",
                description="Research and gather comprehensive information on a given topic",
                agent_id=research_agent_id,
                expected_output="A detailed research report with key findings and analysis",
                dependencies=[]
            ),
            TaskBase(
                id=str(uuid.uuid4()),
                name="Write Content",
                description="Create well-written content based on research findings",
                agent_id=writer_agent_id,
                expected_output="A well-structured piece of content (article, blog post, etc.)",
                dependencies=[]
            )
        ]
        
        tasks.extend(base_tasks)
        
        logger.info(f"✅ Initialized {len(base_agents)} base agents and {len(base_tasks)} base tasks for testing")

def validate_task(task: TaskBase):
    if not task.name:
        raise HTTPException(status_code=400, detail="Name is required")
    
    if not task.description:
        raise HTTPException(status_code=400, detail="Description is required")

    if not task.agent_id:
        raise HTTPException(status_code=400, detail="Agent ID is required")
    
    if not task.expected_output:
        raise HTTPException(status_code=400, detail="Expected output is required")
    
    if task.dependencies is None:
        task.dependencies = []

    return task


@app.get("/tasks", response_model=List[TaskBase])
def get_tasks():
    return tasks

@app.post("/tasks", response_model=TaskBase)
def create_task(task: TaskBase):
    try:
        task = validate_task(task)
        if task.id is None:
            task.id = str(uuid.uuid4())
        tasks.append(task)
        return task
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/tasks/{task_id}", response_model=TaskBase)
def update_task(task_id: str, task_update: TaskBase):
    for i, existing_task in enumerate(tasks):
        if existing_task.id == task_id:
            task_update.id = task_id
            task_update = validate_task(task_update)
            tasks[i] = task_update
            return task_update
    raise HTTPException(status_code=404, detail="Task not found")

@app.delete("/tasks/{task_id}", response_model=TaskBase)
def delete_task(task_id: str):
    for i, task in enumerate(tasks):
        if task.id == task_id:
            deleted_task = tasks.pop(i)
            return deleted_task
    raise HTTPException(status_code=404, detail="Task not found")


@app.get("/tasks/{task_id}", response_model=TaskBase)
def get_task(task_id: str):
    for task in tasks:
        if task.id == task_id:
            return task
    raise HTTPException(status_code=404, detail="Task not found")

# Initialize base data on startup (after all models are defined)
initialize_base_data()


class LaunchTaskConfig(BaseModel):
    task_id: str
    agent_id: str

class LaunchTaskResponse(BaseModel):
    task_id: str
    agent_id: str
    result: str
    status: str

@app.post("/launch-task", response_model=LaunchTaskResponse)
def launch_task(config: LaunchTaskConfig):
    try:
        task = get_task(config.task_id)
        agent_data = get_agent(config.agent_id)
        
        # Create CrewAI Agent
        # Note: tools need to be actual tool objects, not strings
        # For now, we'll create an agent without tools if tools list is empty
        # In a real implementation, you'd need to map tool names to actual tool objects
        crew_agent = Agent(
            role=agent_data.role,
            goal=agent_data.goal,
            backstory=agent_data.backstory,
            tools=[],  # TODO: Map tool names to actual tool objects
            verbose=agent_data.verbose,
            allow_delegation=agent_data.allow_delegation,
            llm=LlamaStackLLM(model_name=DEFAULT_OLLAMA_MODEL)
        )
        
        # Handle dependencies - find dependent tasks
        # For dependencies, we need to get their agents too
        all_tasks = []
        context_tasks = []
        
        if task.dependencies:
            for dep_id in task.dependencies:
                try:
                    dep_task = get_task(dep_id)
                    # Get the agent for the dependency task
                    dep_agent_data = get_agent(dep_task.agent_id)
                    dep_crew_agent = Agent(
                        role=dep_agent_data.role,
                        goal=dep_agent_data.goal,
                        backstory=dep_agent_data.backstory,
                        tools=[],
                        verbose=dep_agent_data.verbose,
                        allow_delegation=dep_agent_data.allow_delegation,
                        llm=LlamaStackLLM(model_name=DEFAULT_OLLAMA_MODEL)
                    )
                    # Create a CrewAI Task for the dependency
                    dep_crew_task = Task(
                        description=dep_task.description,
                        agent=dep_crew_agent,
                        expected_output=dep_task.expected_output
                    )
                    all_tasks.append(dep_crew_task)
                    context_tasks.append(dep_crew_task)
                except HTTPException:
                    logger.warning(f"Dependency task {dep_id} not found, skipping")
        
        # Create CrewAI Task
        crew_task = Task(
            description=task.description,
            agent=crew_agent,
            expected_output=task.expected_output,
            context=context_tasks if context_tasks else None
        )
        all_tasks.append(crew_task)
        
        # Collect all unique agents
        all_agents = [crew_agent]
        for task_obj in all_tasks:
            if task_obj.agent not in all_agents:
                all_agents.append(task_obj.agent)
        
        # Create and run the crew
        crew = Crew(
            agents=all_agents,
            tasks=all_tasks,
            verbose=True
        )
        
        # Execute the crew
        result = crew.kickoff()
        
        return LaunchTaskResponse(
            task_id=task.id,
            agent_id=agent_data.id,
            result=str(result),
            status="completed"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error launching task: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to launch task: {str(e)}") 