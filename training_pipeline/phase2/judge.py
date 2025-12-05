"""
LLM Judge Module

Module để gọi LLM judge đánh giá chất lượng câu trả lời từ 3 methods.
"""

import json
import time
from typing import Dict, Any
from openai import OpenAI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


class LLMJudge:
    """
    LLM Judge để đánh giá câu trả lời từ 3 methods.
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.JUDGE_MODEL
        self.temperature = config.JUDGE_TEMPERATURE
        self.max_tokens = config.JUDGE_MAX_TOKENS
    
    def build_prompt(
        self,
        query: str,
        ground_truth: str,
        context: str,
        answer_1: str,
        answer_2: str,
        answer_3: str
    ) -> str:
        """
        Tạo prompt cho LLM judge.
        
        Args:
            query: Câu hỏi
            ground_truth: Câu trả lời đúng
            context: Context được sử dụng
            answer_1: Câu trả lời từ method 1 (baseline)
            answer_2: Câu trả lời từ method 2 (adaptive mini)
            answer_3: Câu trả lời từ method 3 (adaptive n5)
        
        Returns:
            Prompt string
        """
        prompt = f"""Bạn là một giám khảo chuyên nghiệp đánh giá chất lượng câu trả lời trong hệ thống RAG.

Nhiệm vụ: Đánh giá 3 câu trả lời cho cùng 1 câu hỏi dựa trên ground truth và context được cung cấp.

Câu hỏi: {query}

Ground Truth (câu trả lời đúng): {ground_truth}

Context (ngữ cảnh liên quan): {context}

Câu trả lời cần đánh giá:
1. Method 1 (Baseline - k=5 cố định): {answer_1}

2. Method 2 (Adaptive Mini - k thích ứng n=1): {answer_2}

3. Method 3 (Adaptive N5 - k thích ứng n=5): {answer_3}

Tiêu chí đánh giá (thang điểm 0-1):
- Factual Accuracy: Độ chính xác về mặt sự thật so với ground truth (0.0 = hoàn toàn sai, 1.0 = hoàn toàn đúng)
- Completeness: Độ đầy đủ thông tin so với ground truth (0.0 = thiếu nhiều, 1.0 = đầy đủ)
- Relevance: Độ liên quan đến context được cung cấp (0.0 = không liên quan, 1.0 = rất liên quan)

QUAN TRỌNG về overall_score:
- Nếu câu trả lời SAI về mặt sự thật so với ground truth (factual incorrect): overall_score phải < 0.5
- Nếu câu trả lời ĐÚNG về mặt sự thật (factual correct): overall_score phải >= 0.5
- Score trong khoảng 0.5-1.0 thể hiện mức độ chính xác và đầy đủ (càng cao càng tốt)
- Score trong khoảng 0.0-0.5 chỉ dành cho câu trả lời sai hoặc không liên quan

Với mỗi method, hãy đánh giá 3 tiêu chí trên và tính overall_score là trung bình cộng của 3 tiêu chí, đảm bảo tuân thủ quy tắc trên về ngưỡng 0.5.

Trả về kết quả DƯỚI DẠNG JSON với format sau (chỉ trả về JSON, không có text khác):
{{
  "scores": {{
    "method_1": {{
      "factual_accuracy": 0.85,
      "completeness": 0.80,
      "relevance": 0.90,
      "overall_score": 0.85
    }},
    "method_2": {{
      "factual_accuracy": 0.90,
      "completeness": 0.85,
      "relevance": 0.95,
      "overall_score": 0.90
    }},
    "method_3": {{
      "factual_accuracy": 0.95,
      "completeness": 0.90,
      "relevance": 0.95,
      "overall_score": 0.93
    }}
  }}
}}
"""
        return prompt
    
    def evaluate(
        self,
        query: str,
        ground_truth: str,
        context: str,
        answer_1: str,
        answer_2: str,
        answer_3: str
    ) -> Dict[str, Any]:
        """
        Gọi LLM judge để đánh giá 3 câu trả lời.
        
        Args:
            query: Câu hỏi
            ground_truth: Câu trả lời đúng
            context: Context được sử dụng
            answer_1: Câu trả lời từ method 1
            answer_2: Câu trả lời từ method 2
            answer_3: Câu trả lời từ method 3
        
        Returns:
            Dict với scores và metadata
        """
        prompt = self.build_prompt(query, ground_truth, context, answer_1, answer_2, answer_3)
        
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"}
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        output = response.choices[0].message.content or "{}"
        
        try:
            scores = json.loads(output)
        except json.JSONDecodeError:
            scores = {"error": "Failed to parse JSON response"}
        
        return {
            "scores": scores.get("scores", {}),
            "judge_metadata": {
                "judge_model": self.model,
                "judge_tokens": response.usage.total_tokens,
                "judge_latency_ms": latency_ms,
                "judge_input_tokens": response.usage.prompt_tokens,
                "judge_output_tokens": response.usage.completion_tokens
            }
        }

