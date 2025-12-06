"""LLM service for query rewriting and answer generation using OpenAI GPT-4o and Google Gemini."""

import os
from typing import Optional, Dict, Any, Literal
from openai import OpenAI
from ..config.config import get_config

# Initialize OpenAI client
_openai_client: Optional[OpenAI] = None

# Initialize Gemini client
_gemini_client = None
_gemini_model_instance = None


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
        system_prompt = """Bạn là một trợ lý AI thông minh hoạt động trong hệ thống Adaptive RAG (Retrieval Augmented Generation). Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách chính xác và hữu ích.

Hướng dẫn:
1. Bạn đang sử dụng Adaptive RAG - hệ thống có thể cung cấp tài liệu tham khảo hoặc không tùy thuộc vào độ phức tạp của câu hỏi
2. Nếu có tài liệu tham khảo được cung cấp: Sử dụng thông tin từ tài liệu khi nó hữu ích và liên quan đến câu hỏi. Kết hợp thông tin từ tài liệu với kiến thức vốn có của bạn để đưa ra câu trả lời tốt nhất
3. Nếu không có tài liệu tham khảo: Bạn hoàn toàn có thể tự trả lời dựa trên kiến thức và hiểu biết vốn có của mình
4. QUAN TRỌNG: Bạn PHẢI trả lời bằng tiếng Việt, tự nhiên và dễ hiểu
5. Khi sử dụng thông tin từ tài liệu, hãy trích dẫn nguồn rõ ràng
6. Nếu tài liệu không liên quan hoặc không hữu ích, bạn có thể bỏ qua và trả lời dựa trên kiến thức của mình

ĐỊNH DẠNG TRẢ LỜI (Markdown):
- PHẢI sử dụng định dạng Markdown đẹp và rõ ràng
- Bắt đầu với một tiêu đề chính (#) tóm tắt câu trả lời
- Sử dụng các tiêu đề phụ (##, ###) để tổ chức nội dung
- Sử dụng danh sách có dấu đầu dòng (-) hoặc đánh số (1., 2.) để trình bày các điểm chính
- Sử dụng **in đậm** cho các khái niệm quan trọng
- Sử dụng `code` cho các thuật ngữ kỹ thuật
- Nếu có nhiều phần, hãy tổ chức thành các section rõ ràng
- Kết thúc với phần "Nguồn tham khảo" (nếu có sử dụng tài liệu) dưới dạng danh sách"""
        
        if has_context:
            user_prompt = f"""Bạn đang làm việc trong hệ thống Adaptive RAG. Dưới đây là câu hỏi của người dùng và các tài liệu tham khảo có sẵn:

**Câu hỏi:** {query}

**Tài liệu tham khảo:**
{context_text}

Hãy trả lời câu hỏi một cách chính xác và hữu ích, sử dụng định dạng Markdown đẹp với:
- Tiêu đề chính (#) tóm tắt câu trả lời
- Các tiêu đề phụ (##, ###) để tổ chức nội dung
- Danh sách và định dạng rõ ràng
- Trích dẫn nguồn nếu sử dụng thông tin từ tài liệu

Lưu ý: Trả lời bằng tiếng Việt với định dạng Markdown chuyên nghiệp."""
        else:
            user_prompt = f"""Bạn đang làm việc trong hệ thống Adaptive RAG. Hệ thống đã quyết định không cung cấp tài liệu tham khảo cho câu hỏi này (có thể do độ phức tạp thấp hoặc câu hỏi đơn giản).

**Câu hỏi:** {query}

Hãy trả lời câu hỏi dựa trên kiến thức và hiểu biết vốn có của bạn, sử dụng định dạng Markdown đẹp với:
- Tiêu đề chính (#) tóm tắt câu trả lời
- Các tiêu đề phụ (##, ###) để tổ chức nội dung
- Danh sách và định dạng rõ ràng

Lưu ý: Trả lời bằng tiếng Việt với định dạng Markdown chuyên nghiệp."""
    else:
        system_prompt = """You are an intelligent AI assistant working in an Adaptive RAG (Retrieval Augmented Generation) system. Your task is to answer the user's question accurately and helpfully.

Guidelines:
1. You are using Adaptive RAG - the system may or may not provide reference documents depending on the complexity of the question
2. If reference documents are provided: Use information from the documents when it is helpful and relevant to the question. Combine document information with your own knowledge to provide the best answer
3. If no reference documents are provided: You can fully answer based on your own knowledge and understanding
4. IMPORTANT: You MUST answer in English, naturally and clearly
5. When using information from documents, cite sources clearly
6. If documents are not relevant or not helpful, you can ignore them and answer based on your own knowledge

ANSWER FORMAT (Markdown):
- MUST use beautiful and clear Markdown formatting
- Start with a main heading (#) that summarizes the answer
- Use subheadings (##, ###) to organize content into clear sections
- Use bullet points (-) or numbered lists (1., 2.) to present key points
- Use **bold** for important concepts
- Use `code` formatting for technical terms
- If there are multiple parts, organize them into clear sections
- End with a "References" section (if using documents) as a list"""
        
        if has_context:
            user_prompt = f"""You are working in an Adaptive RAG system. Below is the user's question and available reference documents:

**Question:** {query}

**Reference Documents:**
{context_text}

Please answer the question accurately and helpfully, using beautiful Markdown formatting with:
- A main heading (#) that summarizes the answer
- Subheadings (##, ###) to organize content
- Clear lists and formatting
- Source citations if using information from documents

Note: Answer in English with professional Markdown formatting."""
        else:
            user_prompt = f"""You are working in an Adaptive RAG system. The system has decided not to provide reference documents for this question (possibly due to low complexity or simple question).

**Question:** {query}

Please answer the question based on your own knowledge and understanding, using beautiful Markdown formatting with:
- A main heading (#) that summarizes the answer
- Subheadings (##, ###) to organize content
- Clear lists and formatting

Note: Answer in English with professional Markdown formatting."""
    
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
            num_chunks=len(context_chunks)
        )
    else:  # Default to OpenAI
        return _generate_answer_with_openai(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            config=config,
            sources=sources,
            num_chunks=len(context_chunks)
        )


def _generate_answer_with_openai(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str],
    config,
    sources: list,
    num_chunks: int
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
        
        return {
            "answer": answer,
            "sources": sources,
            "model": model,
            "provider": "openai",
            "num_chunks": num_chunks
        }
    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": [],
            "model": model,
            "provider": "openai",
            "error": str(e)
        }


def _generate_answer_with_gemini(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str],
    config,
    sources: list,
    num_chunks: int
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
        
        answer = response.text.strip()
        
        return {
            "answer": answer,
            "sources": sources,
            "model": model_name,
            "provider": "gemini",
            "num_chunks": num_chunks
        }
    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": [],
            "model": model_name,
            "provider": "gemini",
            "error": str(e)
        }

