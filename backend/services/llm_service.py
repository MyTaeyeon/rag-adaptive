"""LLM service for query rewriting and answer generation using OpenAI GPT models."""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from openai import OpenAI
from ..config.config import get_config
from ..config.prompt import (
    get_query_rewrite_system_prompt,
    get_query_rewrite_user_prompt,
    get_answer_generation_system_prompt,
    get_answer_generation_user_prompt,
    get_logprobs_system_prompt,
    get_answer_generation_system_prompt_with_memory,
    get_answer_generation_user_prompt_with_history
)

# Initialize OpenAI client
_openai_client: Optional[OpenAI] = None

# Log directory for generation prompts/outputs
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_openai_client() -> OpenAI:
    """Get or create OpenAI client instance."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def rewrite_query(
    original_query: str,
    language: str = "en",
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Rewrite user query to improve retrieval performance.
    
    This function uses LLM to expand, clarify, or reformulate the query
    to better match document content.
    
    Parameters
    ----------
    original_query : str
        Original user query
    language : str
        Language code: "en" or "vi"
    model : Optional[str]
        OpenAI model to use. If None, uses config default.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - rewritten_query: str - The rewritten query
        - reasoning: Optional[str] - Explanation of changes (if requested)
    """
    config = get_config()
    if model is None:
        model = config.llm.query_rewrite_model
    
    client = get_openai_client()
    
    # Get prompts from prompt.py
    system_prompt = get_query_rewrite_system_prompt(language)
    user_prompt = get_query_rewrite_user_prompt(original_query, language)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=config.llm.query_rewrite_temperature,
            # max_tokens=config.llm.query_rewrite_max_tokens
            # max_completion_tokens=config.llm.query_rewrite_max_tokens
        )
        
        rewritten_query = response.choices[0].message.content.strip()
        
        return {
            "rewritten_query": rewritten_query,
            "original_query": original_query,
            "model": model
        }
    except Exception as e:
        # Fallback: return original query if rewriting fails
        return {
            "rewritten_query": original_query,
            "original_query": original_query,
            "error": str(e)
        }


def generate_with_logprobs(
    query: str,
    style_prompt: str,
    language: str = "en",
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate response with logprobs for entropy calculation (non-hop call).
    
    This is used in Phase 1 of adaptive RAG to measure model uncertainty
    without retrieved context.
    
    Parameters
    ----------
    query : str
        Original user query
    style_prompt : str
        Style prompt combining query with style instruction
    language : str
        Language code
    model : Optional[str]
        Model to use. If None, uses config default.
    temperature : Optional[float]
        Temperature. If None, uses config default.
    max_tokens : Optional[int]
        Max tokens. If None, uses config default.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - output: Generated text
        - logprobs: List of token logprobs
    """
    config = get_config()
    if model is None:
        model = config.adaptive.phase1_model
    if temperature is None:
        temperature = config.adaptive.phase1_temperature
    if max_tokens is None:
        max_tokens = config.adaptive.phase1_max_tokens
    
    client = get_openai_client()
    
    # Get system prompt from prompt.py
    system_prompt = get_logprobs_system_prompt(language)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": style_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,  # Request logprobs
            top_logprobs=1  # Get top 1 logprob per token
        )
        
        output = response.choices[0].message.content.strip()
        
        # Extract logprobs from response
        logprobs = []
        if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
            content_tokens = response.choices[0].logprobs.content
            if content_tokens:
                for token_info in content_tokens:
                    if token_info.logprob is not None:
                        logprobs.append(token_info.logprob)
        
        return {
            "output": output,
            "logprobs": logprobs,
            "model": model
        }
    except Exception as e:
        return {
            "output": f"Error: {str(e)}",
            "logprobs": [],
            "error": str(e)
        }


def generate_answer(
    query: str,
    context_chunks: list,
    language: str = "en",
    model: Optional[str] = None,
    conversation_history: Optional[str] = None,
    user_memory: Optional[str] = None,
    user_queries_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate answer using retrieved context chunks with OpenAI.
    
    Parameters
    ----------
    query : str
        Original user query
    context_chunks : list
        List of retrieved context chunks (each is a dict with 'text' and 'metadata')
    language : str
        Language code: "en" or "vi"
    model : Optional[str]
        Model name to use. If None, uses config default.
    conversation_history : Optional[str]
        Formatted conversation history to include in prompt
    user_memory : Optional[str]
        Formatted user memory/preferences to include in system prompt
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - answer: str - Generated answer
        - sources: list - List of source metadata
        - model: str - Model used
        - provider: str - Provider used (always "openai")
    """
    config = get_config()
    
    # Prepare context from chunks
    has_context = len(context_chunks) > 0
    context_text = "\n\n".join([
        f"[Document {i+1} - Source: {chunk.get('metadata', {}).get('source', 'Unknown')}]\n{chunk.get('text', '')}"
        for i, chunk in enumerate(context_chunks)
    ]) if has_context else ""
    
    # Get prompts from prompt.py (with memory, user queries context, and history if provided)
    if user_memory or user_queries_context:
        system_prompt = get_answer_generation_system_prompt_with_memory(
            user_memory_text=user_memory or "",
            language=language,
            user_queries_context=user_queries_context
        )
    else:
        system_prompt = get_answer_generation_system_prompt(language)
    
    if conversation_history:
        user_prompt = get_answer_generation_user_prompt_with_history(
            query=query,
            context_text=context_text,
            has_context=has_context,
            conversation_history=conversation_history,
            language=language
        )
    else:
        user_prompt = get_answer_generation_user_prompt(
            query=query,
            context_text=context_text,
            has_context=has_context,
            language=language
        )
    
    # Extract sources from chunks
    sources = [
        {
            "source": chunk.get("metadata", {}).get("source", "Unknown"),
            "chunk_index": chunk.get("metadata", {}).get("chunk_index", -1),
            "score": chunk.get("score", 0.0)
        }
        for chunk in context_chunks
    ]
    
    # Generate answer using OpenAI
    return _generate_answer_with_openai(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        config=config,
        sources=sources,
        num_chunks=len(context_chunks),
        language=language
    )


def _generate_answer_with_openai(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str],
    config,
    sources: list,
    num_chunks: int,
    language: str
) -> Dict[str, Any]:
    """Generate answer using OpenAI."""
    if model is None:
        model = config.llm.answer_generation_model
    
    # Validate model name is not empty
    if not model or not model.strip():
        error_msg = f"Invalid model name: '{model}'. Please check your config."
        error_result = {
            "answer": f"Error: {error_msg}",
            "sources": [],
            "model": model or "unknown",
            "provider": "openai",
            "error": error_msg
        }
        _log_generation(
            provider="openai",
            model=model or "unknown",
            language=language,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            answer="",
            error=error_msg
        )
        return error_result
    
    client = get_openai_client()
    
    try:
        # Build request parameters
        # Note: We don't set temperature or max_tokens for any model to avoid conflicts
        # Some models (like GPT-5) only support default temperature (1) and may error with custom values
        # Some models (like GPT-5) tend to generate very long responses and may error if max_tokens is set
        request_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        }
        
        # Optional: Uncomment below if you want to set temperature/max_tokens for specific models
        # is_gpt5 = config.llm.is_gpt5_model(model)
        # if not is_gpt5:
        #     request_params["temperature"] = config.llm.answer_generation_temperature
        #     request_params["max_completion_tokens"] = config.llm.answer_generation_max_tokens
        
        response = client.chat.completions.create(**request_params)
        
        # Check if response has choices
        if not response.choices or len(response.choices) == 0:
            error_msg = "No choices in response from OpenAI API"
            error_result = {
                "answer": f"Error: {error_msg}",
                "sources": [],
                "model": model,
                "provider": "openai",
                "error": error_msg
            }
            _log_generation(
                provider="openai",
                model=model,
                language=language,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                answer="",
                error=error_msg
            )
            return error_result
        
        # Get the first choice
        choice = response.choices[0]
        
        # Get content first - we need to check this before finish_reason
        content = getattr(choice.message, 'content', None)
        
        # Check finish_reason
        finish_reason = getattr(choice, 'finish_reason', None)
        
        # Handle finish_reason: "length" is NOT an error - it just means response was truncated
        # We should still return the content if it exists
        # Only treat these as actual errors: "content_filter", "safety", etc.
        error_finish_reasons = ['content_filter', 'safety', 'recitation']
        if finish_reason in error_finish_reasons:
            error_msg = f"Response blocked by finish_reason: {finish_reason}"
            error_result = {
                "answer": f"Error: {error_msg}. The response was blocked by content filters.",
                "sources": [],
                "model": model,
                "provider": "openai",
                "error": error_msg,
                "finish_reason": finish_reason
            }
            _log_generation(
                provider="openai",
                model=model,
                language=language,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                answer="",
                error=error_msg
            )
            return error_result
        
        # If finish_reason is "length", it means response was truncated but still valid
        # We'll return the content with a warning if needed
        is_truncated = (finish_reason == "length")
        
        # Check if content is None or empty
        if content is None:
            error_msg = "Response content is None. This may indicate the model returned an empty response."
            error_result = {
                "answer": f"Error: {error_msg}",
                "sources": [],
                "model": model,
                "provider": "openai",
                "error": error_msg,
                "finish_reason": finish_reason
            }
            _log_generation(
                provider="openai",
                model=model,
                language=language,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                answer="",
                error=error_msg
            )
            return error_result
        
        answer = content.strip() if content else ""
        
        # Check if answer is empty after stripping
        if not answer:
            # If finish_reason is "length" but content is empty, it's a real error
            if is_truncated:
                error_msg = f"Response was truncated (finish_reason: length) but content is empty. Consider increasing max_completion_tokens (current: {config.llm.answer_generation_max_tokens})."
            else:
                error_msg = "Response content is empty after stripping. This may indicate the model returned whitespace only."
            error_result = {
                "answer": f"Error: {error_msg}",
                "sources": [],
                "model": model,
                "provider": "openai",
                "error": error_msg,
                "finish_reason": finish_reason
            }
            _log_generation(
                provider="openai",
                model=model,
                language=language,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                answer="",
                error=error_msg
            )
            return error_result
        
        # If truncated, add a note to the answer (optional - you can remove this if you don't want to show it to users)
        # For now, we'll just log it but return the answer normally
        if is_truncated:
            # Log warning but still return the answer
            import logging
            logging.warning(f"Response was truncated (finish_reason: length). Consider increasing max_completion_tokens from {config.llm.answer_generation_max_tokens}.")
        
        result = {
            "answer": answer,
            "sources": sources,
            "model": model,
            "provider": "openai",
            "num_chunks": num_chunks,
            "truncated": is_truncated  # Add flag to indicate if response was truncated
        }
        _log_generation(
            provider="openai",
            model=model,
            language=language,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            answer=answer
        )
        return result
    except Exception as e:
        error_result = {
            "answer": f"Error generating answer: {str(e)}",
            "sources": [],
            "model": model,
            "provider": "openai",
            "error": str(e)
        }
        _log_generation(
            provider="openai",
            model=model,
            language=language,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            answer="",
            error=str(e)
        )
        return error_result
def _log_generation(
    provider: str,
    model: str,
    language: str,
    system_prompt: str,
    user_prompt: str,
    answer: str,
    error: Optional[str] = None
) -> None:
    """Persist prompts and outputs for answer generation step."""
    try:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        # One file per generation with pretty JSON for easy viewing
        log_path = LOG_DIR / f"generation_{timestamp}.json"
        record = {
            "timestamp": timestamp,
            "provider": provider,
            "model": model,
            "language": language,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "answer": answer,
            "error": error,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except Exception:
        # Fail silently to avoid impacting main flow
        pass

