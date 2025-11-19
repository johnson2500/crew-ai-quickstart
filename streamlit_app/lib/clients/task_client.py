from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from .base_client import BaseClient


class TaskBase(BaseModel):
    """Pydantic model for Task data validation."""
    id: Optional[str] = None
    name: str = Field(..., min_length=1, description="The name of the task")
    description: str = Field(..., min_length=1, description="The description of the task")
    agent_id: str = Field(..., min_length=1, description="The ID of the agent assigned to this task")
    expected_output: str = Field(..., min_length=1, description="The expected output of the task")
    dependencies: List[str] = Field(default_factory=list, description="List of task IDs this task depends on")
    
    @field_validator('name', 'description', 'agent_id', 'expected_output')
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Validate that required string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()
    
    @field_validator('agent_id')
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        """Validate that agent_id is a valid UUID format (basic check)."""
        if not v or not v.strip():
            raise ValueError("Agent ID cannot be empty")
        return v.strip()


class TaskClient(BaseClient):
    """Client for task-related API operations."""
    
    def get_tasks(self) -> List[TaskBase]:
        """
        Get all tasks.
        
        Returns:
            List of TaskBase objects
        """
        response = self._get("/tasks")
        return [TaskBase(**task) for task in response]
    
    def get_task(self, task_id: str) -> TaskBase:
        """
        Get a specific task by ID.
        
        Args:
            task_id: The ID of the task to retrieve
            
        Returns:
            TaskBase object
            
        Raises:
            requests.HTTPError: If task not found (404) or other error
        """
        response = self._get(f"/tasks/{task_id}")
        return TaskBase(**response)
    
    def create_task(self, task: TaskBase) -> TaskBase:
        """
        Create a new task.
        
        Args:
            task: TaskBase object with task data (id will be auto-generated if not provided)
            
        Returns:
            Created TaskBase object
        """
        # Convert to dict, excluding None id
        data = task.model_dump(exclude_none=True)
        response = self._post("/tasks", data)
        return TaskBase(**response)
    
    def update_task(self, task_id: str, task: TaskBase) -> TaskBase:
        """
        Update an existing task.
        
        Args:
            task_id: The ID of the task to update
            task: TaskBase object with updated data
            
        Returns:
            Updated TaskBase object
            
        Raises:
            requests.HTTPError: If task not found (404) or other error
        """
        # Ensure the ID matches
        task.id = task_id
        data = task.model_dump(exclude_none=True)
        response = self._put(f"/tasks/{task_id}", data)
        return TaskBase(**response)
    
    def delete_task(self, task_id: str) -> TaskBase:
        """
        Delete a task.
        
        Args:
            task_id: The ID of the task to delete
            
        Returns:
            Deleted TaskBase object
            
        Raises:
            requests.HTTPError: If task not found (404) or other error
        """
        response = self._delete(f"/tasks/{task_id}")
        return TaskBase(**response)

