"""User memory storage for persistent preferences and facts."""

from typing import Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict, field


@dataclass
class UserMemory:
    """
    Stores user preferences and facts.
    
    This is separate from conversation history as it represents
    long-lived information that should persist across conversations.
    """
    preferred_name: Optional[str] = None
    language_preference: Optional[str] = None  # "en" or "vi"
    tone_preference: Optional[str] = None  # "formal" or "casual"
    custom_preferences: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "preferred_name": self.preferred_name,
            "language_preference": self.language_preference,
            "tone_preference": self.tone_preference,
            "custom_preferences": self.custom_preferences,
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "UserMemory":
        """Create from dictionary."""
        return cls(
            preferred_name=data.get("preferred_name"),
            language_preference=data.get("language_preference"),
            tone_preference=data.get("tone_preference"),
            custom_preferences=data.get("custom_preferences", {}),
            last_updated=datetime.fromisoformat(data.get("last_updated", datetime.utcnow().isoformat()))
        )
    
    def format_for_prompt(self, language: str = "en") -> str:
        """
        Format user memory as a string for prompt injection.
        
        Parameters
        ----------
        language : str
            Language for formatting ("en" or "vi")
            
        Returns
        -------
        str
            Formatted memory string
        """
        if language == "vi":
            parts = []
            if self.preferred_name:
                parts.append(f"Người dùng muốn được gọi là: {self.preferred_name}")
            if self.language_preference:
                parts.append(f"Ngôn ngữ ưa thích: {self.language_preference}")
            if self.tone_preference:
                parts.append(f"Phong cách giao tiếp: {self.tone_preference}")
            if self.custom_preferences:
                for key, value in self.custom_preferences.items():
                    parts.append(f"{key}: {value}")
            
            if not parts:
                return ""
            
            return "### THÔNG TIN NGƯỜI DÙNG\n" + "\n".join(f"- {p}" for p in parts)
        else:  # English
            parts = []
            if self.preferred_name:
                parts.append(f"User prefers to be called: {self.preferred_name}")
            if self.language_preference:
                parts.append(f"Language preference: {self.language_preference}")
            if self.tone_preference:
                parts.append(f"Communication tone: {self.tone_preference}")
            if self.custom_preferences:
                for key, value in self.custom_preferences.items():
                    parts.append(f"{key}: {value}")
            
            if not parts:
                return ""
            
            return "### USER PROFILE / MEMORY\n" + "\n".join(f"- {p}" for p in parts)
    
    def is_empty(self) -> bool:
        """Check if memory is empty."""
        return (
            self.preferred_name is None and
            self.language_preference is None and
            self.tone_preference is None and
            len(self.custom_preferences) == 0
        )


class UserMemoryStore:
    """
    In-memory storage for user memories.
    
    Maps session_id -> UserMemory
    In production, this could be persisted to file/DB.
    """
    
    def __init__(self):
        """Initialize user memory store."""
        # session_id -> UserMemory
        self.memories: Dict[str, UserMemory] = {}
    
    def get_memory(self, session_id: str) -> UserMemory:
        """
        Get or create memory for a session.
        
        Parameters
        ----------
        session_id : str
            Session ID
            
        Returns
        -------
        UserMemory
            User memory (new if doesn't exist)
        """
        if session_id not in self.memories:
            self.memories[session_id] = UserMemory()
        return self.memories[session_id]
    
    def update_memory(
        self,
        session_id: str,
        preferred_name: Optional[str] = None,
        language_preference: Optional[str] = None,
        tone_preference: Optional[str] = None,
        **custom_preferences
    ) -> None:
        """
        Update user memory.
        
        Parameters
        ----------
        session_id : str
            Session ID
        preferred_name : Optional[str]
            Preferred name to update
        language_preference : Optional[str]
            Language preference to update
        tone_preference : Optional[str]
            Tone preference to update
        **custom_preferences
            Custom preference key-value pairs
        """
        memory = self.get_memory(session_id)
        
        if preferred_name is not None:
            memory.preferred_name = preferred_name
        if language_preference is not None:
            memory.language_preference = language_preference
        if tone_preference is not None:
            memory.tone_preference = tone_preference
        
        if custom_preferences:
            memory.custom_preferences.update(custom_preferences)
        
        memory.last_updated = datetime.utcnow()
    
    def set_custom_preference(self, session_id: str, key: str, value: Any) -> None:
        """
        Set a custom preference.
        
        Parameters
        ----------
        session_id : str
            Session ID
        key : str
            Preference key
        value : Any
            Preference value
        """
        memory = self.get_memory(session_id)
        memory.custom_preferences[key] = value
        memory.last_updated = datetime.utcnow()
    
    def delete_memory(self, session_id: str) -> bool:
        """
        Delete memory for a session.
        
        Parameters
        ----------
        session_id : str
            Session ID
            
        Returns
        -------
        bool
            True if deleted, False if not found
        """
        if session_id in self.memories:
            del self.memories[session_id]
            return True
        return False


# Global user memory store instance
_memory_store: Optional[UserMemoryStore] = None


def get_user_memory_store() -> UserMemoryStore:
    """Get or create global user memory store instance."""
    global _memory_store
    if _memory_store is None:
        _memory_store = UserMemoryStore()
    return _memory_store

