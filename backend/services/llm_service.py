"""LLM service for query rewriting and answer generation using OpenAI GPT-4o and Google Gemini."""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from openai import OpenAI
from ..config.config import get_config

# Initialize OpenAI client
_openai_client: Optional[OpenAI] = None

# Initialize Gemini client
_gemini_client = None
_gemini_model_instance = None

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


def get_gemini_model(model_name: Optional[str] = None):
    """
    Get or create Gemini model instance.
    
    Parameters
    ----------
    model_name : Optional[str]
        Specific model name to use. If None, uses config default.
        If different from current instance, creates new instance.
    """
    global _gemini_model_instance, _gemini_client
    try:
        import google.generativeai as genai
        _gemini_client = genai
        
        config = get_config()
        api_key = config.llm.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=api_key)
        
        # Use provided model_name or config default
        target_model_name = model_name or config.llm.gemini_model
        
        # Create new instance if needed or if model name changed
        if _gemini_model_instance is None:
            _gemini_model_instance = genai.GenerativeModel(target_model_name)
        elif hasattr(_gemini_model_instance, '_model_name'):
            # Check if model name changed (if we can access it)
            # Otherwise, just recreate for simplicity
            _gemini_model_instance = genai.GenerativeModel(target_model_name)
        else:
            # If we can't check, create new instance
            _gemini_model_instance = genai.GenerativeModel(target_model_name)
            
        return _gemini_model_instance
    except ImportError:
        raise ImportError("google-generativeai package is not installed. Install it with: pip install google-generativeai")


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
    
    # Create prompt based on language
    if language == "vi":
        system_prompt = """Bạn là một chuyên gia tìm kiếm thông tin. Nhiệm vụ của bạn là viết lại truy vấn người dùng để cải thiện hiệu suất tìm kiếm.

Viết lại truy vấn bằng cách:
1. Giữ nguyên ý nghĩa và mục đích ban đầu
2. Mở rộng các từ khóa quan trọng với các từ đồng nghĩa hoặc liên quan
3. Làm rõ ý định nếu truy vấn mơ hồ
4. Giữ nguyên ngôn ngữ tiếng Việt

Chỉ trả về truy vấn đã viết lại, không giải thích thêm."""
        
        user_prompt = f"Viết lại truy vấn sau để tối ưu cho tìm kiếm:\n\n{original_query}"
    else:
        system_prompt = """You are an expert search information specialist. Your task is to rewrite user queries to improve search performance.

Rewrite the query by:
1. Preserving the original meaning and intent
2. Expanding important keywords with synonyms or related terms
3. Clarifying intent if the query is ambiguous
4. Keeping the same language (English)

Return only the rewritten query, no additional explanation."""
        
        user_prompt = f"Rewrite the following query to optimize for search:\n\n{original_query}"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=config.llm.query_rewrite_temperature,
            max_tokens=config.llm.query_rewrite_max_tokens
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
    
    # Create system prompt based on language
    if language == "vi":
        system_prompt = "Bạn là một trợ lý AI. Hãy trả lời câu hỏi theo yêu cầu về phong cách được chỉ định. QUAN TRỌNG: Bạn PHẢI trả lời bằng tiếng Việt."
    else:
        system_prompt = "You are an AI assistant. Answer the question according to the specified style requirement. IMPORTANT: You MUST answer in English."
    
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
    provider: Optional[Literal["openai", "gemini"]] = None
) -> Dict[str, Any]:
    """
    Generate answer using retrieved context chunks.
    
    Parameters
    ----------
    query : str
        Original user query
    context_chunks : list
        List of retrieved context chunks (each is a dict with 'text' and 'metadata')
    language : str
        Language code: "en" or "vi"
    model : Optional[str]
        Model name to use. If None, uses config default for the provider.
    provider : Optional[Literal["openai", "gemini"]]
        LLM provider to use. If None, uses config default.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - answer: str - Generated answer
        - sources: list - List of source metadata
        - model: str - Model used
        - provider: str - Provider used
    """
    config = get_config()
    
    # Determine provider - use config default (from global PROVIDER) if not specified
    if provider is None:
        provider = config.llm.answer_generation_provider or "openai"
    
    # Prepare context from chunks
    has_context = len(context_chunks) > 0
    context_text = "\n\n".join([
        f"[Document {i+1} - Source: {chunk.get('metadata', {}).get('source', 'Unknown')}]\n{chunk.get('text', '')}"
        for i, chunk in enumerate(context_chunks)
    ]) if has_context else ""
    
    # Create prompt based on language
    if language == "vi":
        system_prompt = """
Vai trò: Bạn là Ara trợ lý ảo thân thiện, đa năng, hỗ trợ hỏi đáp với nguời dùng dựa trên tài liệu được cung cấp. Dùng các tài liệu đính kèm để
trả lời truy vấn người dùng nếu các tài liệu có liên quan tới truy vấn. Nếu tài liệu không liên quan, trả lời bằng kiến thức vốn có. Ngoài ra nếu 
người dùng hỏi về hệ thống chatbot của bạn, hãy dùng thông tin phần Bối cảnh phía duới để trả lời.
Bối cảnh: Hệ thống đầy đủ có tên Adaptive RAG Chatbot System, tự động đánh giá câu hỏi, chọn số lượng tài liệu cần truy xuất và phản hồi thông minh. Pipeline gồm: Query Rewriting (GPT-4o), Adaptive K Selection (entropy), Hybrid Retrieval (BM25 + Dense), Reciprocal Rank Fusion, Cross-Encoder Reranking và Answer Generation (GPT-4o/Gemini). Mỗi tài liệu upload được chia nhỏ bằng semantic chunking để giữ ngữ nghĩa, tạo embeddings và lập chỉ mục cho cả BM25 và Dense. Hệ thống linh hoạt trả lời cả khi không có context nếu câu hỏi đơn giản, nhưng sẽ dùng các tài liệu liên quan khi cần và trích dẫn nguồn rõ ràng. Hỗ trợ song ngữ Anh-Việt, tự ưu tiên ngôn ngữ theo truy vấn mới nhất của người dùng.

Mục tiêu: trả lời đúng trọng tâm, dễ hiểu, không dùng biểu tượng/emoji.

Nguyên tắc:
- Bối cảnh: Chỉ đề cập tới bối cảnh hệ thống khi người dùng yêu cầu, không tự đề cập.
- Ngôn ngữ: trả lời bằng ngôn ngữ từ truy vấn cuối cùng của nguời,nằm ở phần **Câu hỏi:** .
- Giọng điệu thân thiện nhưng chuyên nghiệp.
- Rõ ràng và ngắn gọn: ưu tiên câu ngắn, tóm tắt ý chính trước, chi tiết sau.
- Tránh lan man; chỉ nêu thông tin cần thiết cho câu hỏi.
- Sử dụng tài liệu tham khảo nếu phù hợp; nếu không liên quan có thể bỏ qua và trả lời từ kiến thức của bạn.
- Trích nguồn rõ ràng khi sử dụng tài liệu.
- Không dùng biểu tượng hoặc emoji trong câu trả lời.

Định dạng (Markdown):
- Bắt đầu với việc giải thích và tóm tắt câu trả lời.
- Dùng tiêu đề  nếu cần và danh sách gọn gàng.
- In đậm cho khái niệm quan trọng, dùng `code` cho thuật ngữ kỹ thuật.
- Nếu có nguồn, thêm mục \"Nguồn tham khảo\" ở cuối (dạng danh sách)."""
        
        if has_context:
            user_prompt = f""". Dưới đây là câu hỏi của người dùng và các tài liệu tham khảo có sẵn:

**Câu hỏi:** {query}

**Tài liệu tham khảo:**
{context_text}

Hãy trả lời câu hỏi một cách chính xác và hữu ích, sử dụng định dạng Markdown đẹp với:
- Bắt đầu với việc giải thích và tóm tắt câu trả lời.
- Các tiêu đề  nếu cần để tổ chức nội dung
- Danh sách và định dạng rõ ràng
- Trích dẫn nguồn nếu sử dụng thông tin từ tài liệu

Lưu ý: Trả lời bằng ngôn ngữ nguời dùng nhập với định dạng Markdown chuyên nghiệp."""
        else:
            user_prompt = f"""Hệ thống đã quyết định không cung cấp tài liệu tham khảo cho câu hỏi này (có thể do độ phức tạp thấp hoặc câu hỏi đơn giản).

**Câu hỏi:** {query}

Hãy trả lời câu hỏi dựa trên kiến thức và hiểu biết vốn có của bạn, sử dụng định dạng Markdown đẹp với:
- Bắt đầu với việc giải thích và tóm tắt câu trả lời.
- Các tiêu đề nếu cần để tổ chức nội dung
- Danh sách và định dạng rõ ràng

Lưu ý: Trả lời bằng ngôn ngữ nguời dùng nhập với định dạng Markdown chuyên nghiệp."""
    else:
        system_prompt = """Role: You are Ara, a friendly and versatile virtual assistant that answers user questions using the provided documents. Use the attached documents when they are relevant to the query; if not, rely on your own knowledge. If the user asks about the chatbot system itself, use the Background section below.

Background: The full system is named Adaptive RAG Chatbot System. It automatically analyzes the question, picks how many documents to retrieve, and responds intelligently. The pipeline includes: Query Rewriting (GPT-4o), Adaptive K Selection (entropy), Hybrid Retrieval (BM25 + Dense), Reciprocal Rank Fusion, Cross-Encoder Reranking, and Answer Generation (GPT-4o/Gemini). Each uploaded document is chunked with semantic chunking to preserve meaning, embedded, and indexed for both BM25 and Dense. The system can answer even without context when the question is simple, but will use relevant documents and cite sources when helpful. It supports English and Vietnamese, prioritizing the language of the user’s latest query.

Goal: Be accurate, concise, and avoid icons/emojis.

Guidelines:
- Background: Only mention the system background when the user explicitly asks about it; do not bring it up proactively.
- Language: Respond in the language of the user’s latest query (default to English if unclear).
- Tone: Friendly and professional.
- Be clear and succinct: short sentences, key points first, details after.
- Stay on-topic; only include information needed for the question.
- Use reference documents when relevant; if not, answer from your own knowledge.
- Cite sources clearly when you use documents.
- Do not use icons or emojis.

Formatting (Markdown):
- Begin with a short explanation and summary of the answer.
- Use headings if needed and neat lists.
- Bold important concepts; use `code` for technical terms.
- If sources are used, add a “References” section as a list at the end."""
        
        if has_context:
            user_prompt = f"""You are working in an Adaptive RAG system. Below is the user's question and available reference documents:

**Question:** {query}

**Reference Documents:**
{context_text}

Please answer the question accurately and helpfully, using beautiful Markdown formatting with:
- Start with a brief explanation and summary of the answer.
- Clear lists and formatting.
- Source citations if using information from documents.

Note: Respond in the language of the user's input with professional Markdown formatting."""
        else:
            user_prompt = f"""You are working in an Adaptive RAG system. The system has decided not to provide reference documents for this question (possibly due to low complexity or a simple query).

**Question:** {query}

Please answer the question based on your own knowledge and understanding, using beautiful Markdown formatting with:
- Start with a brief explanation and summary of the answer.
- Clear lists and formatting.

Note: Respond in the language of the user's input with professional Markdown formatting."""
    
    # Extract sources from chunks (common for both providers)
    sources = [
        {
            "source": chunk.get("metadata", {}).get("source", "Unknown"),
            "chunk_index": chunk.get("metadata", {}).get("chunk_index", -1),
            "score": chunk.get("score", 0.0)
        }
        for chunk in context_chunks
    ]
    
    # Generate answer based on provider
    if provider == "gemini":
        return _generate_answer_with_gemini(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            config=config,
            sources=sources,
            num_chunks=len(context_chunks),
            language=language
        )
    else:  # Default to OpenAI
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
    
    client = get_openai_client()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=config.llm.answer_generation_temperature,
            max_tokens=config.llm.answer_generation_max_tokens
        )
        
        answer = response.choices[0].message.content.strip()
        
        result = {
            "answer": answer,
            "sources": sources,
            "model": model,
            "provider": "openai",
            "num_chunks": num_chunks
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


def _generate_answer_with_gemini(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str],
    config,
    sources: list,
    num_chunks: int,
    language: str
) -> Dict[str, Any]:
    """Generate answer using Google Gemini."""
    model_name = model or config.llm.gemini_model
    gemini_model = get_gemini_model(model_name=model_name)
    
    # Combine system prompt and user prompt for Gemini
    # Gemini uses a single prompt structure
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    try:
        # Configure generation parameters
        # Gemini accepts generation_config as a dict or GenerationConfig object
        generation_config = {
            "temperature": config.llm.answer_generation_temperature,
            "max_output_tokens": config.llm.answer_generation_max_tokens,
        }
        
        response = gemini_model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        # Check if response was blocked or has issues
        if not response.candidates or len(response.candidates) == 0:
            return {
                "answer": "Error: No response candidates returned from Gemini API.",
                "sources": [],
                "model": model_name,
                "provider": "gemini",
                "error": "No candidates in response"
            }
        
        candidate = response.candidates[0]
        
        # Check finish_reason
        # 0 = FINISH_REASON_UNSPECIFIED
        # 1 = STOP (normal completion)
        # 2 = MAX_TOKENS (hit max token limit)
        # 3 = SAFETY (blocked by safety filters)
        # 4 = RECITATION (blocked due to recitation)
        # 5 = OTHER
        
        finish_reason = candidate.finish_reason if hasattr(candidate, 'finish_reason') else None
        
        # Handle finish_reason - it can be an enum or int
        finish_reason_value = None
        finish_reason_name = None
        
        if finish_reason is not None:
            # If it's an enum, get its value and name
            if hasattr(finish_reason, 'value'):
                finish_reason_value = finish_reason.value
                finish_reason_name = finish_reason.name
            elif hasattr(finish_reason, 'name'):
                finish_reason_name = finish_reason.name
                finish_reason_value = int(finish_reason) if isinstance(finish_reason, (int, str)) else None
            elif isinstance(finish_reason, int):
                finish_reason_value = finish_reason
                finish_reason_map = {
                    0: "UNSPECIFIED",
                    1: "STOP",
                    2: "MAX_TOKENS", 
                    3: "SAFETY",
                    4: "RECITATION",
                    5: "OTHER"
                }
                finish_reason_name = finish_reason_map.get(finish_reason, "UNKNOWN")
            else:
                finish_reason_name = str(finish_reason)
        
        # Check for safety filter blocking (finish_reason = 3 or SAFETY)
        is_safety_blocked = (finish_reason_value == 3 or 
                            (finish_reason_name and finish_reason_name == "SAFETY") or
                            (isinstance(finish_reason, str) and "SAFETY" in str(finish_reason).upper()))
        
        if is_safety_blocked:  # SAFETY
            safety_ratings = candidate.safety_ratings if hasattr(candidate, 'safety_ratings') else []
            blocked_categories = []
            if safety_ratings:
                for r in safety_ratings:
                    try:
                        category_name = r.category.name if hasattr(r.category, 'name') else str(r.category)
                        prob_name = r.probability.name if hasattr(r.probability, 'name') else str(r.probability)
                        if prob_name in ['MEDIUM', 'HIGH']:
                            blocked_categories.append(category_name)
                    except:
                        pass
            error_msg = f"Response blocked by safety filters. Blocked categories: {', '.join(blocked_categories) if blocked_categories else 'Unknown'}"
            error_result = {
                "answer": f"Xin lỗi, câu trả lời đã bị chặn bởi bộ lọc an toàn của Gemini. Lý do: {error_msg}",
                "sources": [],
                "model": model_name,
                "provider": "gemini",
                "error": error_msg
            }
            _log_generation(
                provider="gemini",
                model=model_name,
                language=language,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                answer="",
                error=error_msg
            )
            return error_result
        
        # Check for recitation blocking (finish_reason = 4 or RECITATION)
        is_recitation_blocked = (finish_reason_value == 4 or 
                                (finish_reason_name and finish_reason_name == "RECITATION") or
                                (isinstance(finish_reason, str) and "RECITATION" in str(finish_reason).upper()))
        
        if is_recitation_blocked:  # RECITATION
            error_result = {
                "answer": "Xin lỗi, câu trả lời đã bị chặn do vi phạm chính sách về trích dẫn (recitation policy).",
                "sources": [],
                "model": model_name,
                "provider": "gemini",
                "error": "Response blocked by recitation policy"
            }
            _log_generation(
                provider="gemini",
                model=model_name,
                language=language,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                answer="",
                error="Response blocked by recitation policy"
            )
            return error_result
        
        # Try to extract text from response
        # Note: MAX_TOKENS finish_reason is acceptable - we can still extract partial text
        answer = None
        try:
            if candidate.content and candidate.content.parts and len(candidate.content.parts) > 0:
                part = candidate.content.parts[0]
                if hasattr(part, 'text') and part.text:
                    answer = part.text.strip()
        except (AttributeError, IndexError, TypeError) as e:
            # Log error but continue to check finish_reason
            pass
        
        # If we couldn't extract text, check why
        if not answer:
            # MAX_TOKENS means response was truncated - this is acceptable if we have partial text
            # But if no text at all, it's an error
            if finish_reason_value == 2 or (finish_reason_name and finish_reason_name == "MAX_TOKENS"):
                error_result = {
                    "answer": f"Error: Response was truncated due to max_output_tokens limit ({config.llm.answer_generation_max_tokens}). No content was generated. Please increase max_tokens in config or simplify the query.",
                    "sources": [],
                    "model": model_name,
                    "provider": "gemini",
                    "error": f"Response truncated at max_output_tokens limit with no content, finish_reason: MAX_TOKENS"
                }
                _log_generation(
                    provider="gemini",
                    model=model_name,
                    language=language,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    answer="",
                    error="Response truncated at max_output_tokens limit with no content"
                )
                return error_result
            else:
                error_result = {
                    "answer": f"Error: No content parts in response. Finish reason: {finish_reason_name or finish_reason_value or 'Unknown'}",
                    "sources": [],
                    "model": model_name,
                    "provider": "gemini",
                    "error": f"No content parts, finish_reason: {finish_reason_name or finish_reason_value or finish_reason}"
                }
                _log_generation(
                    provider="gemini",
                    model=model_name,
                    language=language,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    answer="",
                    error=f"No content parts, finish_reason: {finish_reason_name or finish_reason_value or finish_reason}"
                )
                return error_result
        
        result = {
            "answer": answer,
            "sources": sources,
            "model": model_name,
            "provider": "gemini",
            "num_chunks": num_chunks
        }
        _log_generation(
            provider="gemini",
            model=model_name,
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
            "model": model_name,
            "provider": "gemini",
            "error": str(e)
        }
        _log_generation(
            provider="gemini",
            model=model_name,
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

