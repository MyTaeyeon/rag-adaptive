"""Adaptive controller for determining optimal k value using entropy-based method."""

from typing import Literal, Tuple, Dict, Optional, List
from ..config.config import get_config
from ..utils.entropy import sequence_entropy_from_token_logprobs, entropy_to_k
from ..services.llm_service import generate_with_logprobs

Language = Literal["en", "vi"]


def _get_style_by_language(style: str, language: str = "en") -> str:
    """
    Get style instruction in the correct language.
    
    Args:
        style: Original style instruction (usually Vietnamese from config)
        language: Target language code
    
    Returns:
        Style instruction in the target language
    """
    if language == "vi":
        return style
    else:
        # Translate style to English if needed, or use English styles
        english_styles = [
            "Answer briefly and concisely",
            "Answer in detail and comprehensively",
            "Answer in academic style",
            "Answer in friendly style",
            "Answer with specific examples",
            "Answer in list format",
            "Answer step by step",
            "Focus on core concepts",
            "Answer with comparison and contrast",
            "Answer with historical context"
        ]
        # Use corresponding English style
        config = get_config()
        try:
            style_idx = config.adaptive.style_candidates.index(style) if style in config.adaptive.style_candidates else 0
            if style_idx < len(english_styles):
                return english_styles[style_idx]
        except (AttributeError, ValueError):
            pass
        # Fallback to first English style if style not found
        return english_styles[0] if english_styles else style


def _build_style_prompt(query: str, style: str, language: str = "en") -> str:
    """
    Build style prompt by combining query with style instruction.
    
    Args:
        query: User query
        style: Style instruction (will be converted to correct language)
        language: Language code
    
    Returns:
        Style prompt string
    """
    style_in_language = _get_style_by_language(style, language)
    return f"{style_in_language}:\n\n{query}"


class AdaptiveController:
    """
    Adaptive controller using entropy-based method.
    
    Performs n iterations with different response styles to calculate entropy
    and determine optimal k value for retrieval.
    """

    def __init__(self, k_min: Optional[int] = None, k_max: Optional[int] = None) -> None:
        config = get_config()
        self.k_min = k_min if k_min is not None else config.adaptive.k_min
        self.k_max = k_max if k_max is not None else config.adaptive.k_max
        self.config = config

    def decide(
        self, 
        query: str, 
        language: Language = "en",
        n: Optional[int] = None
    ) -> Tuple[int, Dict]:
        """
        Decide optimal k value using adaptive entropy-based method.
        
        Phase 1: Run n iterations with different styles to calculate entropy.
        Each iteration calls LLM without context (non-hop) to measure uncertainty.
        
        Parameters
        ----------
        query : str
            User query
        language : Language
            Language code
        n : Optional[int]
            Number of iterations (1-10). If None, uses config default.
            
        Returns
        -------
        Tuple[int, Dict]
            (k_value, metadata) where metadata contains:
            - average_entropy: Average entropy from n iterations
            - k_determined: Determined k value
            - iterations_detail: List of iteration details
            - n: Number of iterations
        """
        if n is None:
            n = self.config.adaptive.default_n
        n = max(1, min(10, n))  # Ensure n is in range 1-10
        
        iterations_detail = []
        entropies = []
        
        # Phase 1: Run n iterations with different styles
        for i in range(n):
            # Select style (cycle through if n > len(styles))
            style = self.config.adaptive.style_candidates[i % len(self.config.adaptive.style_candidates)]
            style_prompt = _build_style_prompt(query, style, language)
            
            # Generate with logprobs (non-hop call)
            gen_result = generate_with_logprobs(
                query=query,
                style_prompt=style_prompt,
                language=language,
                model=self.config.adaptive.phase1_model,
                temperature=self.config.adaptive.phase1_temperature,
                max_tokens=self.config.adaptive.phase1_max_tokens
            )
            
            # Calculate entropy from logprobs
            logprobs = gen_result.get("logprobs", [])
            if logprobs:
                entropy = sequence_entropy_from_token_logprobs(logprobs)
            else:
                entropy = 0.0
            
            entropies.append(entropy)
            
            # Store iteration details with style in correct language
            style_in_language = _get_style_by_language(style, language)
            iterations_detail.append({
                "run": i + 1,
                "style": style_in_language,
                "output": gen_result.get("output", ""),
                "entropy": entropy
            })
        
        # Calculate average entropy and determine k
        average_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        k_determined = entropy_to_k(
            average_entropy, 
            k_min=self.k_min, 
            k_max=self.k_max,
            entropy_max=self.config.adaptive.entropy_max
        )
        
        metadata = {
            "average_entropy": average_entropy,
            "k_determined": k_determined,
            "iterations_detail": iterations_detail,
            "n": n,
            "k_min": self.k_min,
            "k_max": self.k_max
        }
        
        return k_determined, metadata

