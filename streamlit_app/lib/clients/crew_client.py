from typing import Optional
from pydantic import BaseModel, Field, field_validator
from .base_client import BaseClient


class LaunchTaskConfig(BaseModel):
    """Pydantic model for launch task configuration."""
    task_id: str = Field(..., min_length=1, description="The ID of the task to launch")
    agent_id: str = Field(..., min_length=1, description="The ID of the agent to use for the task")
    
    @field_validator('task_id', 'agent_id')
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Validate that required string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class LaunchTaskResponse(BaseModel):
    """Pydantic model for launch task response."""
    task_id: str = Field(..., description="The ID of the task that was launched")
    agent_id: str = Field(..., description="The ID of the agent that executed the task")
    result: str = Field(..., description="The result of the task execution")
    status: str = Field(..., description="The status of the task execution")


class CrewClient(BaseClient):
    """Client for crew-related API operations."""
    
    def launch_task(self, config: LaunchTaskConfig) -> LaunchTaskResponse:
        """
        Launch a task with a specific agent.
        
        Args:
            config: LaunchTaskConfig object with task_id and agent_id
            
        Returns:
            LaunchTaskResponse object with execution results
            
        Raises:
            requests.HTTPError: If task or agent not found, or execution fails
        """
        data = config.model_dump()
        response = self._post("/launch-task", data)
        return LaunchTaskResponse(**response)
    
    def launch_task_by_ids(self, task_id: str, agent_id: str) -> LaunchTaskResponse:
        """
        Launch a task with a specific agent using IDs directly.
        
        Args:
            task_id: The ID of the task to launch
            agent_id: The ID of the agent to use
            
        Returns:
            LaunchTaskResponse object with execution results
            
        Raises:
            requests.HTTPError: If task or agent not found, or execution fails
        """
        config = LaunchTaskConfig(task_id=task_id, agent_id=agent_id)
        return self.launch_task(config)

