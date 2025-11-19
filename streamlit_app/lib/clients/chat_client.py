from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from .base_client import BaseClient


class ChatRequest(BaseModel):
    """Pydantic model for chat request validation."""
    message: str = Field(..., min_length=1, description="The user's message")
    model: str = Field(..., min_length=1, description="The model identifier to use")
    session_id: Optional[str] = Field(default=None, description="Optional session ID for conversation continuity")
    use_rag: bool = Field(default=False, description="Whether to use RAG (Retrieval-Augmented Generation)")
    
    @field_validator('message', 'model')
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Validate that required string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class ChatResponse(BaseModel):
    """Pydantic model for chat response validation."""
    response: str = Field(..., description="The assistant's response")
    session_id: str = Field(..., description="The session ID for this conversation")


class IngestResponse(BaseModel):
    """Pydantic model for ingest response validation."""
    status: str = Field(..., description="Status of the ingestion (success, skipped, etc.)")
    files_processed: int = Field(..., ge=0, description="Number of files processed")
    details: str = Field(..., description="Details about the ingestion process")


class ChatClient(BaseClient):
    """Client for chat-related API operations."""
    
    def get_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of model identifier strings
        """
        response = self._get("/models")
        # Handle both list and dict responses
        if isinstance(response, list):
            return response
        elif isinstance(response, dict) and "data" in response:
            return [m.get("identifier", m) if isinstance(m, dict) else m for m in response["data"]]
        else:
            return []
    
    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send a chat message and get a response.
        
        Args:
            request: ChatRequest object with message, model, and optional session_id and use_rag
            
        Returns:
            ChatResponse object with response and session_id
        """
        data = request.model_dump(exclude_none=True)
        response = self._post("/chat", data)
        return ChatResponse(**response)
    
    def chat_message(
        self,
        message: str,
        model: str,
        session_id: Optional[str] = None,
        use_rag: bool = False
    ) -> ChatResponse:
        """
        Send a chat message using individual parameters.
        
        Args:
            message: The user's message
            model: The model identifier to use
            session_id: Optional session ID for conversation continuity
            use_rag: Whether to use RAG
            
        Returns:
            ChatResponse object with response and session_id
        """
        request = ChatRequest(
            message=message,
            model=model,
            session_id=session_id,
            use_rag=use_rag
        )
        return self.chat(request)
    
    def ingest_knowledge(self) -> IngestResponse:
        """
        Ingest documents from the knowledge folder into the vector database.
        
        Returns:
            IngestResponse object with status, files_processed, and details
        """
        response = self._post("/ingest", {})
        return IngestResponse(**response)
    
    def get_files_in_vector_db(self) -> List[str]:
        """
        Get list of files in the vector database.
        
        Returns:
            List of filenames in the vector database
        """
        response = self._get("/files-in-vector-db")
        if isinstance(response, list):
            return response
        return []

