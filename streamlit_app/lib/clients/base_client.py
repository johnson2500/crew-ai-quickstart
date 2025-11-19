import os
import requests
from typing import Optional, Dict, Any


class BaseClient:
    """Base client for making HTTP requests to the FastAPI backend."""
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the base client.
        
        Args:
            base_url: Base URL for the FastAPI server. Defaults to http://localhost:8000
        """
        self.base_url = base_url or os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")
        # Remove trailing slash if present
        self.base_url = self.base_url.rstrip("/")
    
    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a GET request to the API.
        
        Args:
            endpoint: API endpoint (e.g., "/agents")
            params: Optional query parameters
            
        Returns:
            JSON response as dictionary
            
        Raises:
            requests.HTTPError: If the request fails
        """
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a POST request to the API.
        
        Args:
            endpoint: API endpoint (e.g., "/agents")
            data: Request body as dictionary
            
        Returns:
            JSON response as dictionary
            
        Raises:
            requests.HTTPError: If the request fails
        """
        url = f"{self.base_url}{endpoint}"
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def _put(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a PUT request to the API.
        
        Args:
            endpoint: API endpoint (e.g., "/agents/{agent_id}")
            data: Request body as dictionary
            
        Returns:
            JSON response as dictionary
            
        Raises:
            requests.HTTPError: If the request fails
        """
        url = f"{self.base_url}{endpoint}"
        response = requests.put(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def _delete(self, endpoint: str) -> Dict[str, Any]:
        """
        Make a DELETE request to the API.
        
        Args:
            endpoint: API endpoint (e.g., "/agents/{agent_id}")
            
        Returns:
            JSON response as dictionary
            
        Raises:
            requests.HTTPError: If the request fails
        """
        url = f"{self.base_url}{endpoint}"
        response = requests.delete(url)
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the API.
        
        Returns:
            Health status dictionary
        """
        return self._get("/health")

