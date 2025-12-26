"""Conversation history storage for chat messages."""

from typing import List, Dict, Optional, Literal
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class Message:
    """Represents a single message in conversation history."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


class ConversationHistory:
    """
    In-memory conversation history storage.
    
    Stores messages per session with sliding window support
    to avoid prompt bloat.
    """
    
    def __init__(self, max_messages: int = 20):
        """
        Initialize conversation history storage.
        
        Parameters
        ----------
        max_messages : int
            Maximum messages to keep per session (sliding window)
        """
        # session_id -> List[Message]
        self.histories: Dict[str, List[Message]] = {}
        self.max_messages = max_messages
    
    def add_message(
        self,
        session_id: str,
        role: Literal["user", "assistant", "system"],
        content: str
    ) -> None:
        """
        Add a message to conversation history.
        
        Parameters
        ----------
        session_id : str
            Session ID
        role : Literal["user", "assistant", "system"]
            Message role
        content : str
            Message content
        """
        if session_id not in self.histories:
            self.histories[session_id] = []
        
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.utcnow()
        )
        
        self.histories[session_id].append(message)
        
        # Apply sliding window
        if len(self.histories[session_id]) > self.max_messages:
            # Keep the most recent messages
            self.histories[session_id] = self.histories[session_id][-self.max_messages:]
    
    def get_recent_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        include_system: bool = False
    ) -> List[Message]:
        """
        Get recent messages for a session.
        
        Parameters
        ----------
        session_id : str
            Session ID
        limit : Optional[int]
            Maximum number of messages to return (None = all)
        include_system : bool
            Whether to include system messages
            
        Returns
        -------
        List[Message]
            List of recent messages
        """
        if session_id not in self.histories:
            return []
        
        messages = self.histories[session_id]
        
        # Filter system messages if needed
        if not include_system:
            messages = [msg for msg in messages if msg.role != "system"]
        
        # Apply limit
        if limit is not None:
            messages = messages[-limit:]
        
        return messages
    
    def get_all_messages(self, session_id: str) -> List[Message]:
        """
        Get all messages for a session.
        
        Parameters
        ----------
        session_id : str
            Session ID
            
        Returns
        -------
        List[Message]
            All messages in the session
        """
        return self.histories.get(session_id, [])
    
    def format_for_prompt(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
        include_system: bool = False
    ) -> str:
        """
        Format conversation history as a string for prompt injection.
        
        Parameters
        ----------
        session_id : str
            Session ID
        max_messages : Optional[int]
            Maximum messages to include (None = all)
        include_system : bool
            Whether to include system messages
            
        Returns
        -------
        str
            Formatted conversation history
        """
        messages = self.get_recent_messages(
            session_id,
            limit=max_messages,
            include_system=include_system
        )
        
        if not messages:
            return ""
        
        formatted = []
        for msg in messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            formatted.append(f"{role_label}: {msg.content}")
        
        return "\n".join(formatted)
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all messages for a session.
        
        Parameters
        ----------
        session_id : str
            Session ID
            
        Returns
        -------
        bool
            True if cleared, False if session not found
        """
        if session_id in self.histories:
            del self.histories[session_id]
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """Alias for clear_session for consistency."""
        return self.clear_session(session_id)
    
    def get_user_queries(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Get recent user queries (questions) only.
        
        Parameters
        ----------
        session_id : str
            Session ID
        limit : Optional[int]
            Maximum number of queries to return (None = all)
            
        Returns
        -------
        List[str]
            List of user query strings (content only)
        """
        if session_id not in self.histories:
            return []
        
        # Filter only user messages
        user_messages = [
            msg for msg in self.histories[session_id]
            if msg.role == "user"
        ]
        
        # Extract content
        queries = [msg.content for msg in user_messages]
        
        # Apply limit (get most recent N queries)
        if limit is not None and limit > 0:
            queries = queries[-limit:]
        
        return queries


# Global conversation history instance
_history: Optional[ConversationHistory] = None


def get_conversation_history() -> ConversationHistory:
    """Get or create global conversation history instance."""
    global _history
    if _history is None:
        _history = ConversationHistory()
    return _history

