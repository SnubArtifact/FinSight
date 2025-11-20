import logging

import google.generativeai as genai

from utils.config import GEMINI_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


def _normalize_model_name(model_name: str) -> str:
    """Google SDK accepts bare model ids (no 'models/' prefix)."""
    if model_name.startswith("models/"):
        stripped = model_name.split("/", 1)[-1]
        logger.warning(
            "LLM_MODEL contained 'models/' prefix (%s); using '%s' instead.",
            model_name,
            stripped,
        )
        return stripped
    return model_name


genai.configure(api_key=GEMINI_API_KEY)
_model_name = _normalize_model_name(LLM_MODEL)
logger.info("Using Gemini chat model: %s", _model_name)
model = genai.GenerativeModel(_model_name)

def generate_answer(query: str, context_chunks: list, conversation_history: list = None):
    """Generate answer using RAG with optional conversation history.
    
    Args:
        query: Current user question
        context_chunks: Retrieved document chunks
        conversation_history: List of (question, answer) tuples from previous turns
    """
    context = "\n\n".join(context_chunks)
    
    # Build conversation context if history exists
    history_text = ""
    if conversation_history:
        history_text = "\n\nPrevious conversation:\n"
        for i, (prev_q, prev_a) in enumerate(conversation_history, 1):
            history_text += f"\nQ{i}: {prev_q}\nA{i}: {prev_a}\n"

    prompt = f"""You are Finsight AI, a financial analysis assistant. You are knowledgeable, helpful, and analytical.

INSTRUCTIONS:
- Use the document context and conversation history to answer questions
- Connect information across different parts of the document when relevant
- Make reasonable inferences based on available data
- If information isn't explicitly stated, provide insights based on what IS available
- NEVER say "the context doesn't contain information" or apologize for missing data
- Instead, work with what you have and provide the best possible answer
- For follow-up questions, reference previous answers naturally
- Be conversational and helpful, not robotic

Document Context:
{context}
{history_text}

Current Query:
{query}

Provide a helpful, analytical answer. Connect the dots between different pieces of information when relevant.
"""
    try:
        response = model.generate_content(prompt)
    except Exception as exc:
        logger.error("Gemini generate_content failed: %s", exc, exc_info=True)
        raise

    if not getattr(response, "text", "").strip():
        logger.warning("Gemini response contained no text. Raw response: %s", response)
        return "I couldn't generate an answer. Please try rephrasing."

    return response.text.strip()
