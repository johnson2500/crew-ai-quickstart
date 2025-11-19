import os
from typing import Dict, List

LLAMA_STACK_URL = os.getenv("LLAMA_STACK_URL", "http://localhost:8321")
# Knowledge folder is now in API/knowledge
KNOWLEDGE_FOLDER = os.path.join(os.path.dirname(__file__), "knowledge")
VECTOR_DB_ID = "vs_0eb3e18c-553f-4254-b0b2-9dd830ea1146"
# In-memory session storage
SESSION_STORAGE: Dict[str, List[Dict[str, str]]] = {}