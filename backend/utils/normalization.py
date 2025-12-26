"""Text normalization utilities for English and Vietnamese."""

import re
from typing import Literal

try:
    import unicodedata
    UNICODEDATA_AVAILABLE = True
except ImportError:
    UNICODEDATA_AVAILABLE = False

Language = Literal["en", "vi"]


def english_normalize(text: str) -> str:
    """
    Normalize English text by converting to lowercase.
    
    This function provides minimal normalization suitable for English retrieval.
    It only converts text to lowercase to perform case-insensitive matching.
    """
    return text.lower()


def vietnamese_normalize(text: str) -> str:
    """
    Normalize Vietnamese text by removing diacritics and converting to lowercase.
    
    This function removes Vietnamese diacritics (accents) to improve retrieval
    for Vietnamese text by making the search accent-insensitive.
    """
    if not UNICODEDATA_AVAILABLE:
        # Fallback: just lowercase if unicodedata not available
        return text.lower()
    
    # Remove diacritics
    nfd = unicodedata.normalize('NFD', text)
    normalized = ''.join(
        char for char in nfd 
        if unicodedata.category(char) != 'Mn'
    )
    # Convert to lowercase
    return normalized.lower()


def normalize_text(text: str, language: Language = "en") -> str:
    """
    Normalize text based on language.
    
    Parameters
    ----------
    text : str
        Input text to normalize
    language : Language
        Language code: "en" for English, "vi" for Vietnamese
        
    Returns
    -------
    str
        Normalized text
    """
    if language == "vi":
        return vietnamese_normalize(text)
    else:
        return english_normalize(text)

