"""Conversation API client utilities for Streamlit UI."""
import http
import os
import requests
from typing import Optional, List, Dict, Any
from uuid import UUID
import json

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080/api/v1")


class ConversationClient:
    """Client for interacting with conversation API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
    
    def check_health(self) -> bool:
        """Check if API is reachable."""
        try:
            response = requests.get(f"{self.base_url.replace('/api/v1', '')}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"[ConvClient] Health check failed: {e}")
            return False
    
    def create_conversation(self, title: Optional[str] = None, user_id: Optional[str] = None) -> Optional[Dict]:
        """Create a new conversation."""
        try:
            data = {"title": title or "New Chat", "user_id": user_id}
            print(f"[ConvClient] Creating conversation with data: {data}")
            response = requests.post(f"{self.base_url}/conversations", json=data, timeout=10)
            print(f"[ConvClient] Response status: {response.status_code}")
            response.raise_for_status()
            result = response.json()
            # Convert UUID to string for serialization
            if result and "id" in result:
                result["id"] = str(result["id"])
            print(f"[ConvClient] Created conversation: {result}")
            return result
        except Exception as e:
            print(f"[ConvClient] ERROR creating conversation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def list_conversations(self, user_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get all conversations for a user."""
        try:
            params = {"limit": limit}
            if user_id:
                params["user_id"] = user_id
            print(f"[ConvClient] Fetching conversations with params: {params}")
            response = requests.get(f"{self.base_url}/conversations", params=params, timeout=10)
            print(f"[ConvClient] Response status: {response.status_code}")
            response.raise_for_status()
            result = response.json()
            # Convert UUIDs to strings for serialization
            if isinstance(result, list):
                for item in result:
                    if "id" in item:
                        item["id"] = str(item["id"])
            print(f"[ConvClient] Got {len(result)} conversations")
            return result
        except Exception as e:
            print(f"[ConvClient] ERROR listing conversations: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get a specific conversation with all messages."""
        try:
            response = requests.get(f"{self.base_url}/conversations/{conversation_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting conversation {conversation_id}: {e}")
            return None
    
    def add_message(
        self, 
        conversation_id: str, 
        role: str, 
        content: str, 
        research_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Add a message to a conversation."""
        try:
            data = {
                "role": role,
                "content": content,
                "research_id": research_id,
                "context_data": metadata,
            }
            response = requests.post(
                f"{self.base_url}/conversations/{conversation_id}/messages",
                json=data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error adding message to conversation {conversation_id}: {e}")
            return None
    
    def get_messages(
        self, 
        conversation_id: str, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[Dict]:
        """Get messages from a conversation."""
        try:
            params = {"limit": limit, "offset": offset}
            response = requests.get(
                f"{self.base_url}/conversations/{conversation_id}/messages",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting messages from {conversation_id}: {e}")
            return []
    
    def update_conversation(self, conversation_id: str, title: str) -> Optional[Dict]:
        """Update conversation title."""
        try:
            params = {"title": title}
            response = requests.put(
                f"{self.base_url}/conversations/{conversation_id}",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error updating conversation {conversation_id}: {e}")
            return None
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        try:
            response = requests.delete(f"{self.base_url}/conversations/{conversation_id}")
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Error deleting conversation {conversation_id}: {e}")
            return False
