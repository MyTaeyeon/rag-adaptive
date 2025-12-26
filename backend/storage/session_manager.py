"""Session management for tracking user conversations."""

import uuid
from typing import Dict, Optional
from datetime import datetime, timedelta


class SessionManager:
    """
    Simple in-memory session manager for demo purposes.
    
    Each session represents a conversation with a user.
    Sessions can be extended to persist to file/DB in production.
    """
    
    def __init__(self, session_ttl_hours: int = 24):
        """
        Initialize session manager.
        
        Parameters
        ----------
        session_ttl_hours : int
            Hours until session expires (default: 24)
        """
        self.sessions: Dict[str, Dict] = {}
        self.session_ttl = timedelta(hours=session_ttl_hours)
    
    def create_session(
        self,
        collection_name: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Create a new session.
        
        Parameters
        ----------
        collection_name : Optional[str]
            Associated collection name
        user_id : Optional[str]
            Optional user identifier
            
        Returns
        -------
        str
            Session ID
        """
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "session_id": session_id,
            "collection_name": collection_name,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Get session information.
        
        Parameters
        ----------
        session_id : str
            Session ID
            
        Returns
        -------
        Optional[Dict]
            Session data or None if not found/expired
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if expired
        if datetime.utcnow() - session["created_at"] > self.session_ttl:
            del self.sessions[session_id]
            return None
        
        # Update last activity
        session["last_activity"] = datetime.utcnow()
        return session
    
    def update_session(self, session_id: str, **kwargs) -> bool:
        """
        Update session metadata.
        
        Parameters
        ----------
        session_id : str
            Session ID
        **kwargs
            Fields to update
            
        Returns
        -------
        bool
            True if updated, False if session not found
        """
        if session_id not in self.sessions:
            return False
        
        self.sessions[session_id].update(kwargs)
        self.sessions[session_id]["last_activity"] = datetime.utcnow()
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Parameters
        ----------
        session_id : str
            Session ID
            
        Returns
        -------
        bool
            True if deleted, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def cleanup_expired(self) -> int:
        """
        Remove expired sessions.
        
        Returns
        -------
        int
            Number of sessions removed
        """
        now = datetime.utcnow()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session["created_at"] > self.session_ttl
        ]
        for sid in expired:
            del self.sessions[sid]
        return len(expired)


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager

