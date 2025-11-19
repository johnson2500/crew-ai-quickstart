from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from .base_client import BaseClient


class AgentBase(BaseModel):
    """Pydantic model for Agent data validation."""
    id: Optional[str] = None
    role: str = Field(..., min_length=1, description="The role of the agent")
    goal: str = Field(..., min_length=1, description="The goal of the agent")
    backstory: str = Field(..., min_length=1, description="The backstory of the agent")
    tools: List[str] = Field(default_factory=list, description="List of tool names")
    verbose: bool = Field(default=False, description="Whether the agent should be verbose")
    allow_delegation: bool = Field(default=False, description="Whether the agent can delegate tasks")
    
    @field_validator('role', 'goal', 'backstory')
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Validate that required string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class AgentClient(BaseClient):
    """Client for agent-related API operations."""
    
    def get_agents(self) -> List[AgentBase]:
        """
        Get all agents.
        
        Returns:
            List of AgentBase objects
        """
        response = self._get("/agents")
        return [AgentBase(**agent) for agent in response]
    
    def get_agent(self, agent_id: str) -> AgentBase:
        """
        Get a specific agent by ID.
        
        Args:
            agent_id: The ID of the agent to retrieve
            
        Returns:
            AgentBase object
            
        Raises:
            requests.HTTPError: If agent not found (404) or other error
        """
        response = self._get(f"/agents/{agent_id}")
        return AgentBase(**response)
    
    def create_agent(self, agent: AgentBase) -> AgentBase:
        """
        Create a new agent.
        
        Args:
            agent: AgentBase object with agent data (id will be auto-generated if not provided)
            
        Returns:
            Created AgentBase object
        """
        # Convert to dict, excluding None id
        data = agent.model_dump(exclude_none=True)
        response = self._post("/agents", data)
        return AgentBase(**response)
    
    def update_agent(self, agent_id: str, agent: AgentBase) -> AgentBase:
        """
        Update an existing agent.
        
        Args:
            agent_id: The ID of the agent to update
            agent: AgentBase object with updated data
            
        Returns:
            Updated AgentBase object
            
        Raises:
            requests.HTTPError: If agent not found (404) or other error
        """
        # Ensure the ID matches
        agent.id = agent_id
        data = agent.model_dump(exclude_none=True)
        response = self._put(f"/agents/{agent_id}", data)
        return AgentBase(**response)
    
    def delete_agent(self, agent_id: str) -> AgentBase:
        """
        Delete an agent.
        
        Args:
            agent_id: The ID of the agent to delete
            
        Returns:
            Deleted AgentBase object
            
        Raises:
            requests.HTTPError: If agent not found (404) or other error
        """
        response = self._delete(f"/agents/{agent_id}")
        return AgentBase(**response)

