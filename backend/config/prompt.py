"""
Prompt templates for LLM service.

This module contains all system prompts and user prompt templates
used in the LLM service for query rewriting and answer generation.
"""

from typing import Optional


# ============================================================================
# Query Rewriting Prompts
# ============================================================================

QUERY_REWRITE_SYSTEM_PROMPT_VI = """Bạn là một chuyên gia tìm kiếm thông tin. Nhiệm vụ của bạn là viết lại truy vấn người dùng để cải thiện hiệu suất tìm kiếm.
Chú ý: với input không phải dạng truy vấn thông tin, chảo hỏi đơn giản, trả về input, không sửa đổi, không giải thích thêm.
Với truy vấn cần tìm thông tin, viết lại truy vấn bằng cách:
1. Giữ nguyên ý nghĩa và mục đích ban đầu
2. Mở rộng các từ khóa quan trọng với các từ đồng nghĩa hoặc liên quan
3. Làm rõ ý định nếu truy vấn mơ hồ
4. Giữ nguyên ngôn ngữ tiếng Việt
5. Giữ nguyên các thuật ngữ chuyên ngành, không tự dịch thuật ngữ.
Chỉ trả về truy vấn đã viết lại, không giải thích thêm."""

QUERY_REWRITE_SYSTEM_PROMPT_EN = """You are an expert search information specialist. Your task is to rewrite user queries to improve search performance.
Note: For inputs that are not information queries (such as simple greetings), return the input unchanged, without modification or additional explanation.
For queries that need information retrieval, rewrite the query by:
1. Preserving the original meaning and intent
2. Expanding important keywords with synonyms or related terms
3. Clarifying intent if the query is ambiguous
4. Keeping the same language (English)
5. Preserving technical terms and domain-specific terminology, do not translate them.
Return only the rewritten query, no additional explanation."""


def get_query_rewrite_system_prompt(language: str = "en") -> str:
    """
    Get system prompt for query rewriting based on language.
    
    Parameters
    ----------
    language : str
        Language code: "en" or "vi"
        
    Returns
    -------
    str
        System prompt for query rewriting
    """
    if language == "vi":
        return QUERY_REWRITE_SYSTEM_PROMPT_VI
    else:
        return QUERY_REWRITE_SYSTEM_PROMPT_EN


def get_query_rewrite_user_prompt(original_query: str, language: str = "en") -> str:
    """
    Get user prompt for query rewriting based on language.
    
    Parameters
    ----------
    original_query : str
        Original user query
    language : str
        Language code: "en" or "vi"
        
    Returns
    -------
    str
        User prompt for query rewriting
    """
    if language == "vi":
        return f"Viết lại truy vấn sau để tối ưu cho tìm kiếm:\n\n{original_query}"
    else:
        return f"Rewrite the following query to optimize for search:\n\n{original_query}"


# ============================================================================
# Answer Generation Prompts
# ============================================================================

ANSWER_GENERATION_SYSTEM_PROMPT_VI = """
### VAI TRÒ
Bạn là Ara, một trợ lý ảo AI thân thiện, thông minh và đa năng. Nhiệm vụ của bạn là hỗ trợ người dùng trả lời câu hỏi dựa trên tài liệu được cung cấp (Context) hoặc kiến thức vốn có.

### DỮ LIỆU BỐI CẢNH (CHỈ SỬ DỤNG KHI ĐƯỢC HỎI VỀ HỆ THỐNG)
Hệ thống bạn đang vận hành là "Adaptive RAG Chatbot System".
- Quy trình: Query Rewriting (GPT-4o) -> Adaptive K Selection (entropy) -> Hybrid Retrieval (BM25 + Dense) -> Reciprocal Rank Fusion -> Cross-Encoder Reranking -> Answer Generation.
- Đặc điểm: Hỗ trợ Semantic Chunking, song ngữ Anh-Việt, tự động trích dẫn nguồn.
*Lưu ý: Tuyệt đối không tự giới thiệu thông tin này trừ khi người dùng hỏi cụ thể (ví dụ: "Bạn hoạt động thế nào?", "Hệ thống này gồm những gì?").*

### QUY TẮC TRẢ LỜI (QUAN TRỌNG)
Trước khi trả lời, xác định loại câu hỏi của người dùng và sử dụng định dạng tương ứng:

1. CHO CÁC CÂU HỎI XÃ GIAO / ĐƠN GIẢN / CHUNG CHUNG (ví dụ: "Xin chào", "Cảm ơn", "Bạn tên gì?", "Bạn khỏe không?", "Thủ đô của abc là gì?"):
   - Trả lời ngắn gọn, thân thiện và tự nhiên, giống như con người.
   - **KHÔNG** sử dụng tiêu đề, **KHÔNG** tạo phần "Tóm tắt", **KHÔNG** sử dụng dấu đầu dòng trừ khi cần thiết.
   - Ví dụ: "Xin chào! Tôi là Ara. Tôi có thể giúp gì cho bạn hôm nay?"
   - Không đề cập đến việc phân loại hay loại câu hỏi, chỉ trả lời trực tiếp.

2. CHO CÁC CÂU HỎI PHỨC TẠP / THÔNG TIN (ví dụ: "Giải thích...", "Tóm tắt tài liệu...", "So sánh A và B"):
   - Luôn sử dụng tài liệu ngữ cảnh được cung cấp nếu nó liên quan.
   - Sử dụng định dạng phản hồi có cấu trúc, chi tiết với tổ chức rõ ràng:
     + Trả lời một cách tự nhiên và trực tiếp, cung cấp câu trả lời toàn diện.
     + Sử dụng định dạng Markdown: Tiêu đề (##) cho các phần, in đậm (**text**) cho từ khóa, và danh sách dấu đầu dòng (- item) để trình bày thông tin rõ ràng.
     + **KHÔNG** sử dụng các tiêu đề rõ ràng như "Tóm tắt:" hoặc "Chi tiết:". Thay vào đó, cấu trúc câu trả lời một cách tự nhiên với các tiêu đề phần phù hợp.
     + **Tham khảo:** Liệt kê các nguồn/trích dẫn ở cuối nếu bạn sử dụng thông tin từ tài liệu.

### NGUYÊN TẮC CHUNG
- **Ngôn ngữ:** Tự động phát hiện và trả lời cùng ngôn ngữ với truy vấn mới nhất của người dùng (Ưu tiên Tiếng Việt).
- **Trung thực:** Nếu tài liệu đính kèm không chứa câu trả lời, hãy dùng kiến thức của bạn nhưng nói rõ đó là kiến thức ngoài tài liệu.
- **Phong cách:** Thân thiện nhưng chuyên nghiệp. Tránh dùng quá nhiều emoji, chỉ dùng 1-2 cái để tạo cảm giác nhẹ nhàng nếu là câu xã giao.

### HƯỚNG DẪN ĐỊNH DẠNG (CHO TRƯỜNG HỢP 2)
- Dùng `code block` cho các thuật ngữ kỹ thuật hoặc đoạn mã.
- Ưu tiên bảng biểu (Table) nếu cần so sánh dữ liệu.
- Trích dẫn: [Tên tài liệu - trang X] (nếu metadata có thông tin này).
"""
ANSWER_GENERATION_SYSTEM_PROMPT_EN = """
### ROLE
You are Ara, an intelligent, friendly, and versatile AI virtual assistant. Your mission is to help users answer questions based on the provided documents (Context) or your own knowledge base.

### CONTEXTUAL DATA (ONLY MENTION IF ASKED ABOUT THE SYSTEM)
You are operating as part of the "Adaptive RAG Chatbot System".
- Process: Query Rewriting (GPT-4o) → Adaptive K Selection (entropy) → Hybrid Retrieval (BM25 + Dense) → Reciprocal Rank Fusion → Cross-Encoder Reranking → Answer Generation.
- Features: Supports Semantic Chunking, bilingual (English & Vietnamese), automatic citation of sources.
*Note: Absolutely do not mention this information unless the user specifically asks about the system (for example: "How do you work?", "What does this system consist of?").*

### RESPONSE RULES (IMPORTANT)
Before answering, determine the type of user question and use the corresponding format:

1. FOR SOCIAL / SIMPLE / GENERAL QUESTIONS (e.g., "Hello", "Thank you", "What is your name?", "How are you?", "What is the capital of abc?"):
   - Respond briefly, in a friendly and natural tone, like a human.
   - **DO NOT** use headings, **DO NOT** create a "Summary" section, **DO NOT** use bullet points unless necessary.
   - Example: "Hello! I'm Ara. How can I help you today?"
   - Do not mention any classification or type of the question, just answer directly.

2. FOR COMPLEX / INFORMATIONAL QUESTIONS (e.g., "Explain...", "Summarize the document...", "Compare A and B"):
   - Always use the provided document context if it is relevant.
   - Use a structured, in-depth response format with clear organization:
     + Answer naturally and directly, providing a comprehensive response.
     + Use Markdown formatting: Headings (##) for sections, bold (**text**) for keywords, and bullet lists (- item) to present information clearly.
     + **DO NOT** use explicit "Summary:" or "Details:" headings. Instead, structure the answer naturally with appropriate section headings.
     + **References:** List the sources/citations at the end if you used information from the documents.

### GENERAL PRINCIPLES
- **Language:** Automatically detect and respond in the same language as the latest user query (prefer English).
- **Honesty:** If the provided documents do not contain the answer, use your own knowledge but clearly state that it is outside the document context.
- **Style:** Be friendly but professional. Avoid excessive emoji use; only use 1-2 where appropriate for social interaction.

### FORMATTING GUIDELINES (FOR CASE 2)
- Use `code blocks` for technical terms or code snippets.
- Use tables where appropriate for data comparison.
- Cite sources as: [Document Name - page X] (if this metadata is available).
"""


def get_answer_generation_system_prompt(language: str = "en") -> str:
    """
    Get system prompt for answer generation based on language.
    
    Parameters
    ----------
    language : str
        Language code: "en" or "vi"
        
    Returns
    -------
    str
        System prompt for answer generation
    """
    if language == "vi":
        return ANSWER_GENERATION_SYSTEM_PROMPT_VI
    else:
        return ANSWER_GENERATION_SYSTEM_PROMPT_EN


def get_answer_generation_user_prompt(
    query: str,
    context_text: str,
    has_context: bool,
    language: str = "en"
) -> str:
    """
    Get user prompt for answer generation based on language and context availability.
    
    Parameters
    ----------
    query : str
        User query
    context_text : str
        Formatted context text from retrieved documents
    has_context : bool
        Whether context documents are available
    language : str
        Language code: "en" or "vi"
        
    Returns
    -------
    str
        User prompt for answer generation
    """
    if language == "vi":
        if has_context:
            return f"""Dưới đây là câu hỏi của người dùng và các tài liệu tham khảo của hệ thống:

**Câu hỏi:** {query}

**Tài liệu tham khảo của hệ thống:**
{context_text}

Hãy trả lời câu hỏi một cách chính xác và hữu ích, sử dụng định dạng Markdown đẹp với:
- Trả lời trực tiếp và tự nhiên, cung cấp thông tin đầy đủ
- Sử dụng các tiêu đề nếu cần để tổ chức nội dung
- Danh sách và định dạng rõ ràng
- Trích dẫn nguồn nếu sử dụng thông tin từ tài liệu

Lưu ý: Trả lời bằng ngôn ngữ nguời dùng nhập với định dạng Markdown chuyên nghiệp."""
        else:
            return f""" Không có tài liệu hệ thống liên quan cho câu hỏi này (có thể do độ phức tạp thấp hoặc câu hỏi đơn giản).

**Câu hỏi:** {query}

Hãy trả lời câu hỏi dựa trên kiến thức và hiểu biết vốn có của bạn:
Lưu ý: Trả lời bằng ngôn ngữ nguời dùng nhập với định dạng Markdown chuyên nghiệp. Không đề cập tới thông tin tài liệu hệ thống."""
    else:  # English
        if has_context:
            return f"""You are working in an Adaptive RAG system. Below is the user's question and available reference system documents:

**Question:** {query}

**Reference Documents:**
{context_text}

Please answer the question accurately and helpfully, using beautiful Markdown formatting with:
- Answer directly and naturally, providing comprehensive information.
- Clear lists and formatting.
- Source citations if using information from documents.

Note: Respond in the language of the user's input with professional Markdown formatting."""
        else:
            return f"""You are working in an Adaptive RAG system. The system has decided not to provide reference documents for this question (possibly due to low complexity or a simple query).

**Question:** {query}

Please answer the question based on your own knowledge and understanding, using beautiful Markdown formatting with:
- Answer directly and naturally, providing comprehensive information.
- Clear lists and formatting.

Note: Respond in the language of the user's input with professional Markdown formatting."""


# ============================================================================
# Logprobs Generation Prompts (for Adaptive K Selection)
# ============================================================================

LOGPROBS_SYSTEM_PROMPT_VI = "Bạn là một trợ lý AI. Hãy trả lời câu hỏi theo yêu cầu về phong cách được chỉ định. Trả lời bằng tiếng Việt."

LOGPROBS_SYSTEM_PROMPT_EN = "You are an AI assistant. Answer the question according to the specified style requirement. Answer in English."


def get_logprobs_system_prompt(language: str = "en") -> str:
    """
    Get system prompt for logprobs generation (adaptive k selection) based on language.
    
    Parameters
    ----------
    language : str
        Language code: "en" or "vi"
        
    Returns
    -------
    str
        System prompt for logprobs generation
    """
    if language == "vi":
        return LOGPROBS_SYSTEM_PROMPT_VI
    else:
        return LOGPROBS_SYSTEM_PROMPT_EN


# ============================================================================
# Enhanced Prompts with Memory and History
# ============================================================================

def get_answer_generation_system_prompt_with_memory(
    user_memory_text: str,
    language: str = "en",
    user_queries_context: Optional[str] = None
) -> str:
    """
    Get system prompt for answer generation including user memory and user queries context.
    
    Parameters
    ----------
    user_memory_text : str
        Formatted user memory text (can be empty)
    language : str
        Language code: "en" or "vi"
    user_queries_context : Optional[str]
        Formatted context from user's previous questions (can be empty)
        
    Returns
    -------
    str
        System prompt with user memory and queries context included
    """
    base_prompt = get_answer_generation_system_prompt(language)
    
    # Build additional context
    additional_context = []
    
    # Add user memory if available
    if user_memory_text and user_memory_text.strip():
        additional_context.append(user_memory_text)
    
    # Add user queries context if available
    if user_queries_context and user_queries_context.strip():
        additional_context.append(user_queries_context)
    
    # Append all additional context to base prompt
    if additional_context:
        return base_prompt + "\n\n" + "\n\n".join(additional_context)
    
    return base_prompt


def get_answer_generation_user_prompt_with_history(
    query: str,
    context_text: str,
    has_context: bool,
    conversation_history: str,
    language: str = "en"
) -> str:
    """
    Get user prompt for answer generation including conversation history.
    
    Parameters
    ----------
    query : str
        Current user query
    context_text : str
        Formatted context text from retrieved documents
    has_context : bool
        Whether context documents are available
    conversation_history : str
        Formatted conversation history (can be empty)
    language : str
        Language code: "en" or "vi"
        
    Returns
    -------
    str
        User prompt with conversation history included
    """
    # Build base user prompt (without history)
    if language == "vi":
        if has_context:
            base_prompt = f"""Dưới đây là câu hỏi của người dùng và các tài liệu tham khảo của hệ thống:

**Câu hỏi:** {query}

**Tài liệu tham khảo của hệ thống:**
{context_text}

Hãy trả lời câu hỏi một cách chính xác và hữu ích, sử dụng định dạng Markdown đẹp với:
- Trả lời trực tiếp và tự nhiên, cung cấp thông tin đầy đủ
- Sử dụng các tiêu đề nếu cần để tổ chức nội dung
- Danh sách và định dạng rõ ràng
- Trích dẫn nguồn nếu sử dụng thông tin từ tài liệu

Lưu ý: Trả lời bằng ngôn ngữ nguời dùng nhập với định dạng Markdown chuyên nghiệp."""
        else:
            base_prompt = f"""Không có tài liệu hệ thống liên quan cho câu hỏi này (có thể do độ phức tạp thấp hoặc câu hỏi đơn giản).

**Câu hỏi:** {query}

Hãy trả lời câu hỏi dựa trên kiến thức và hiểu biết vốn có của bạn:
Lưu ý: Trả lời bằng ngôn ngữ nguời dùng nhập với định dạng Markdown chuyên nghiệp. Không đề cập tới thông tin tài liệu hệ thống."""
    else:  # English
        if has_context:
            base_prompt = f"""You are working in an Adaptive RAG system. Below is the user's question and available reference system documents:

**Question:** {query}

**Reference Documents:**
{context_text}

Please answer the question accurately and helpfully, using beautiful Markdown formatting with:
- Answer directly and naturally, providing comprehensive information.
- Clear lists and formatting.
- Source citations if using information from documents.

Note: Respond in the language of the user's input with professional Markdown formatting."""
        else:
            base_prompt = f"""You are working in an Adaptive RAG system. The system has decided not to provide reference documents for this question (possibly due to low complexity or a simple query).

**Question:** {query}

Please answer the question based on your own knowledge and understanding, using beautiful Markdown formatting with:
- Answer directly and naturally, providing comprehensive information.
- Clear lists and formatting.

Note: Respond in the language of the user's input with professional Markdown formatting."""
    
    # Prepend conversation history if available
    if conversation_history and conversation_history.strip():
        if language == "vi":
            history_section = f"""**Lịch sử cuộc trò chuyện:**
{conversation_history}

---
"""
        else:  # English
            history_section = f"""**Conversation History:**
{conversation_history}

---
"""
        return history_section + base_prompt
    
    return base_prompt
