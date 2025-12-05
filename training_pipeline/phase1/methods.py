"""
RAG Methods Implementation

Implement 3 methods: Baseline, Adaptive Mini, Adaptive N5
"""

import sys
from typing import Dict, List
from pathlib import Path
import time
import math
from typing import List as ListType

sys.path.append(str(Path(__file__).parent.parent))
import config
from generation import LLMGenerator
from retrieval import InMemoryRetrieval
from training import sequence_entropy_from_token_logprobs, entropy_to_k, _build_style_prompt


class BaselineRAG:
    """
    Method 1: Baseline RAG with fixed k=5
    """
    
    def __init__(self, retrieval: InMemoryRetrieval, generator: LLMGenerator):
        self.retrieval = retrieval
        self.generator = generator
        self.k = config.BASELINE_K
    
    def run(self, query: str) -> Dict:
        """
        Run baseline RAG method.
        
        Args:
            query: Query text
        
        Returns:
            Tracking structure for method_1_baseline
        """
        start_time = time.time()
        
        chunks = self.retrieval.retrieve(query, k=self.k)
        retrieved_context = "\n\n".join([chunk["text"] for chunk in chunks])
        
        gen_result = self.generator.generate(
            query=query,
            context=retrieved_context,
            model=config.GENERATION_MODEL,
            temperature=config.GENERATION_TEMPERATURE,
            max_tokens=config.GENERATION_MAX_TOKENS
        )
        
        total_latency_ms = (time.time() - start_time) * 1000
        
        return {
            "description": "Fixed k=5",
            "config": {
                "k_fixed": self.k
            },
            "execution_metrics": {
                "latency_ms": total_latency_ms,
                "total_input_tokens": gen_result["total_input_tokens"],
                "total_output_tokens": gen_result["total_output_tokens"]
            },
            "final_output": gen_result["output"],
            "status": "success"
        }


class AdaptiveMiniRAG:
    """
    Method 2: Adaptive RAG Mini (n=1)
    """
    
    def __init__(self, retrieval: InMemoryRetrieval, generator: LLMGenerator):
        self.retrieval = retrieval
        self.generator = generator
        self.k_min = config.K_MIN
        self.k_max = config.K_MAX
    
    def run(self, query: str) -> Dict:
        """
        Run adaptive mini RAG method.
        
        Args:
            query: Query text
        
        Returns:
            Tracking structure for method_2_adaptive_mini
        """
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0
        
        phase1_start = time.time()
        
        gen_result_phase1 = self.generator.generate(
            query=query,
            model=config.ADAPTIVE_PHASE1_MODEL,
            temperature=config.ADAPTIVE_PHASE1_TEMPERATURE,
            max_tokens=config.ADAPTIVE_PHASE1_MAX_TOKENS,
            logprobs=True
        )
        
        total_input_tokens += gen_result_phase1["total_input_tokens"]
        total_output_tokens += gen_result_phase1["total_output_tokens"]
        
        phase1_latency_ms = (time.time() - phase1_start) * 1000
        
        logprobs = gen_result_phase1.get("logprobs", [])
        if logprobs:
            entropy = sequence_entropy_from_token_logprobs(logprobs)
        else:
            entropy = 0.0
        
        k_determined = entropy_to_k(entropy, k_min=self.k_min, k_max=self.k_max)
        
        phase2_start = time.time()
        
        chunks = self.retrieval.retrieve(query, k=k_determined)
        retrieved_context = "\n\n".join([chunk["text"] for chunk in chunks])
        
        gen_result_phase2 = self.generator.generate(
            query=query,
            context=retrieved_context,
            model=config.GENERATION_MODEL,
            temperature=config.GENERATION_TEMPERATURE,
            max_tokens=config.GENERATION_MAX_TOKENS
        )
        
        total_input_tokens += gen_result_phase2["total_input_tokens"]
        total_output_tokens += gen_result_phase2["total_output_tokens"]
        
        phase2_latency_ms = (time.time() - phase2_start) * 1000
        total_latency_ms = (time.time() - start_time) * 1000
        
        return {
            "description": "Adaptive k based on single pass entropy",
            "config": {
                "k_min": self.k_min,
                "k_max": self.k_max
            },
            "execution_metrics": {
                "latency_ms": total_latency_ms,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens
            },
            "phase_1_analysis": {
                "llm_output_raw": gen_result_phase1["output"],
                "calculated_entropy": entropy,
                "k_determined": k_determined
            },
            "final_output": gen_result_phase2["output"],
            "status": "success"
        }


class AdaptiveN5RAG:
    """
    Method 3: Adaptive RAG Full (n=5)
    """
    
    def __init__(self, retrieval: InMemoryRetrieval, generator: LLMGenerator):
        self.retrieval = retrieval
        self.generator = generator
        self.k_min = config.K_MIN
        self.k_max = config.K_MAX
        self.n = config.N_ITERATIONS
    
    def run(self, query: str) -> Dict:
        """
        Run adaptive N5 RAG method.
        
        Args:
            query: Query text
        
        Returns:
            Tracking structure for method_3_adaptive_n5
        """
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0
        
        phase1_start = time.time()
        iterations_detail = []
        entropies = []
        
        for i in range(self.n):
            style = config.STYLE_CANDIDATES[i % len(config.STYLE_CANDIDATES)]
            style_prompt = _build_style_prompt(query, style)
            
            iter_start = time.time()
            
            gen_result = self.generator.generate(
                query=query,
                model=config.ADAPTIVE_PHASE1_MODEL,
                temperature=config.ADAPTIVE_PHASE1_TEMPERATURE,
                max_tokens=config.ADAPTIVE_PHASE1_MAX_TOKENS,
                logprobs=True,
                custom_prompt=style_prompt
            )
            
            iter_latency_ms = (time.time() - iter_start) * 1000
            total_input_tokens += gen_result["total_input_tokens"]
            total_output_tokens += gen_result["total_output_tokens"]
            
            logprobs = gen_result.get("logprobs", [])
            if logprobs:
                entropy = sequence_entropy_from_token_logprobs(logprobs)
            else:
                entropy = 0.0
            
            entropies.append(entropy)
            
            iterations_detail.append({
                "run": i + 1,
                "style": style,
                "output": gen_result["output"],
                "entropy": entropy
            })
        
        average_entropy = sum(entropies) / len(entropies)
        k_determined = entropy_to_k(average_entropy, k_min=self.k_min, k_max=self.k_max)
        
        phase1_latency_ms = (time.time() - phase1_start) * 1000
        
        phase2_start = time.time()
        
        chunks = self.retrieval.retrieve(query, k=k_determined)
        retrieved_context = "\n\n".join([chunk["text"] for chunk in chunks])
        
        gen_result_phase2 = self.generator.generate(
            query=query,
            context=retrieved_context,
            model=config.GENERATION_MODEL,
            temperature=config.GENERATION_TEMPERATURE,
            max_tokens=config.GENERATION_MAX_TOKENS
        )
        
        total_input_tokens += gen_result_phase2["total_input_tokens"]
        total_output_tokens += gen_result_phase2["total_output_tokens"]
        
        phase2_latency_ms = (time.time() - phase2_start) * 1000
        total_latency_ms = (time.time() - start_time) * 1000
        
        return {
            "description": "Adaptive k based on average entropy of 5 passes",
            "config": {
                "n_iterations": self.n,
                "k_min": self.k_min,
                "k_max": self.k_max
            },
            "execution_metrics": {
                "latency_ms": total_latency_ms,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens
            },
            "phase_1_analysis": {
                "n_iterations": self.n,
                "average_entropy": average_entropy,
                "k_determined": k_determined,
                "iterations_detail": iterations_detail
            },
            "final_output": gen_result_phase2["output"],
            "status": "success"
        }

