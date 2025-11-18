from pydantic import BaseModel
from typing import Optional, List

# --- Data Models ---
class ChatRequest(BaseModel):
    message: str
    model: str
    session_id: Optional[str] = None
    use_rag: bool = False

class ChatResponse(BaseModel):
    response: str
    session_id: str

class IngestResponse(BaseModel):
    status: str
    files_processed: int
    details: str

class VerifyResponse(BaseModel):
    found: bool
    query: str
    chunks_found: int
    sample_chunks: List[str]