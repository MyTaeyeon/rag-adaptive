"""Service for extracting user preferences from messages."""

import re
from typing import Dict, Optional, Tuple
from ..storage.user_memory import get_user_memory_store


# Patterns for detecting user preferences (Vietnamese and English)
PREFERRED_NAME_PATTERNS_VI = [
    r"gọi\s+tôi\s+là\s+([^\s\.\?!,]+)",
    r"hãy\s+gọi\s+tôi\s+là\s+([^\s\.\?!,]+)",
    r"tên\s+tôi\s+là\s+([^\s\.\?!,]+)",
    r"gọi\s+tôi\s+([^\s\.\?!,]+)",
    r"tôi\s+là\s+([^\s\.\?!,]+)",
    r"tên\s+của\s+tôi\s+là\s+([^\s\.\?!,]+)",
]

PREFERRED_NAME_PATTERNS_EN = [
    r"call\s+me\s+([^\s\.\?!,]+)",
    r"my\s+name\s+is\s+([^\s\.\?!,]+)",
    r"i\s+am\s+([^\s\.\?!,]+)",
    r"i'm\s+([^\s\.\?!,]+)",
    r"name\s+is\s+([^\s\.\?!,]+)",
]

LANGUAGE_PREFERENCE_PATTERNS_VI = [
    r"trả\s+lời\s+bằng\s+tiếng\s+(việt|anh)",
    r"nói\s+bằng\s+tiếng\s+(việt|anh)",
    r"giao\s+tiếp\s+bằng\s+tiếng\s+(việt|anh)",
]

LANGUAGE_PREFERENCE_PATTERNS_EN = [
    r"respond\s+in\s+(english|vietnamese)",
    r"speak\s+in\s+(english|vietnamese)",
    r"use\s+(english|vietnamese)",
]

TONE_PREFERENCE_PATTERNS_VI = [
    r"giao\s+tiếp\s+(trang\s+trọng|thân\s+thiện)",
    r"phong\s+cách\s+(trang\s+trọng|thân\s+thiện)",
    r"nói\s+chuyện\s+(trang\s+trọng|thân\s+thiện)",
]

TONE_PREFERENCE_PATTERNS_EN = [
    r"be\s+(formal|casual|friendly)",
    r"tone\s+should\s+be\s+(formal|casual)",
    r"speak\s+(formally|casually)",
]


def extract_preferred_name(message: str) -> Optional[str]:
    """
    Extract preferred name from message.
    
    Parameters
    ----------
    message : str
        User message
        
    Returns
    -------
    Optional[str]
        Extracted name or None
    """
    message_lower = message.lower()
    
    # Try Vietnamese patterns
    for pattern in PREFERRED_NAME_PATTERNS_VI:
        match = re.search(pattern, message_lower, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Filter out very short or common words
            if len(name) >= 2 and name not in ["nhé", "đi", "nhá"]:
                return name
    
    # Try English patterns
    for pattern in PREFERRED_NAME_PATTERNS_EN:
        match = re.search(pattern, message_lower, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if len(name) >= 2:
                return name
    
    return None


def extract_language_preference(message: str) -> Optional[str]:
    """
    Extract language preference from message.
    
    Parameters
    ----------
    message : str
        User message
        
    Returns
    -------
    Optional[str]
        "en" or "vi" or None
    """
    message_lower = message.lower()
    
    # Try Vietnamese patterns
    for pattern in LANGUAGE_PREFERENCE_PATTERNS_VI:
        match = re.search(pattern, message_lower, re.IGNORECASE)
        if match:
            lang = match.group(1).lower()
            if "việt" in lang or "vietnamese" in lang:
                return "vi"
            elif "anh" in lang or "english" in lang:
                return "en"
    
    # Try English patterns
    for pattern in LANGUAGE_PREFERENCE_PATTERNS_EN:
        match = re.search(pattern, message_lower, re.IGNORECASE)
        if match:
            lang = match.group(1).lower()
            if "vietnamese" in lang or "việt" in lang:
                return "vi"
            elif "english" in lang or "en" in lang:
                return "en"
    
    return None


def extract_tone_preference(message: str) -> Optional[str]:
    """
    Extract tone preference from message.
    
    Parameters
    ----------
    message : str
        User message
        
    Returns
    -------
    Optional[str]
        "formal" or "casual" or None
    """
    message_lower = message.lower()
    
    # Try Vietnamese patterns
    for pattern in TONE_PREFERENCE_PATTERNS_VI:
        match = re.search(pattern, message_lower, re.IGNORECASE)
        if match:
            tone = match.group(1).lower()
            if "trang trọng" in tone or "formal" in tone:
                return "formal"
            elif "thân thiện" in tone or "casual" in tone:
                return "casual"
    
    # Try English patterns
    for pattern in TONE_PREFERENCE_PATTERNS_EN:
        match = re.search(pattern, message_lower, re.IGNORECASE)
        if match:
            tone = match.group(1).lower()
            if "formal" in tone or "formally" in tone:
                return "formal"
            elif "casual" in tone or "casually" in tone or "friendly" in tone:
                return "casual"
    
    return None


def extract_and_store_preferences(
    session_id: str,
    user_message: str
) -> Dict[str, Optional[str]]:
    """
    Extract user preferences from message and store them.
    
    Parameters
    ----------
    session_id : str
        Session ID
    user_message : str
        User message to analyze
        
    Returns
    -------
    Dict[str, Optional[str]]
        Dictionary of extracted preferences:
        - preferred_name: Optional[str]
        - language_preference: Optional[str]
        - tone_preference: Optional[str]
    """
    memory_store = get_user_memory_store()
    
    # Extract preferences
    preferred_name = extract_preferred_name(user_message)
    language_preference = extract_language_preference(user_message)
    tone_preference = extract_tone_preference(user_message)
    
    # Store if found
    if preferred_name or language_preference or tone_preference:
        memory_store.update_memory(
            session_id=session_id,
            preferred_name=preferred_name,
            language_preference=language_preference,
            tone_preference=tone_preference
        )
    
    return {
        "preferred_name": preferred_name,
        "language_preference": language_preference,
        "tone_preference": tone_preference
    }

