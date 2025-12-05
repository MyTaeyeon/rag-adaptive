"""
LLM Generation Module

Module xử lý việc gọi LLM để generate answers với tracking metrics.
"""

import sys
from typing import Dict, Tuple, Optional
from pathlib import Path
import time
from openai import OpenAI

sys.path.append(str(Path(__file__).parent.parent))
import config


class LLMGenerator:
    """
    LLM Generator với tracking metrics.
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        logprobs: bool = False,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate answer từ LLM với tracking metrics.
        
        Args:
            query: Query text
            context: Optional context để thêm vào prompt
            model: Model name (mặc định từ config)
            temperature: Temperature (mặc định từ config)
            max_tokens: Max tokens (mặc định từ config)
            logprobs: Có lấy logprobs không (cho entropy calculation)
        
        Returns:
            Dict với keys:
            {
                "output": "...",
                "total_input_tokens": 100,
                "total_output_tokens": 50,
                "latency_ms": 1200,
                "entropy": 0.45,  # Chỉ có nếu logprobs=True
                "logprobs": [...]  # Chỉ có nếu logprobs=True
            }
        """
        if model is None:
            model = config.GENERATION_MODEL
        if temperature is None:
            temperature = config.GENERATION_TEMPERATURE
        if max_tokens is None:
            max_tokens = config.GENERATION_MAX_TOKENS
        
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = self._build_prompt(query, context)
        
        start_time = time.time()
        
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 5
        
        response = self.client.chat.completions.create(**kwargs)
        
        latency_ms = (time.time() - start_time) * 1000
        
        choice = response.choices[0]
        output = choice.message.content or ""
        
        result = {
            "output": output,
            "total_input_tokens": response.usage.prompt_tokens,
            "total_output_tokens": response.usage.completion_tokens,
            "latency_ms": latency_ms
        }
        
        if logprobs and choice.logprobs and choice.logprobs.content:
            logprobs_list = [t.logprob for t in choice.logprobs.content if t.logprob is not None]
            result["logprobs"] = logprobs_list
        
        return result
    
    def _build_prompt(self, query: str, context: Optional[str] = None) -> str:
        """
        Build prompt từ query và context.
        
        Args:
            query: Query text
            context: Optional context
        
        Returns:
            Full prompt string
        """
        if context:
            return f"Context:\n{context}\n\nCâu hỏi: {query}\n\nTrả lời:"
        else:
            return f"Trả lời ngắn gọn, trực tiếp:\n\nCâu hỏi: {query}"

