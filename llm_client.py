"""OpenAI LLM Client for NASA Mission Intelligence RAG System.

Provides conversation management and response generation using OpenAI's
chat completion API, with context integration for RAG-based retrieval.
"""

from typing import Dict, List
from openai import OpenAI


# System prompt positioning the assistant as a NASA mission expert
SYSTEM_PROMPT = (
    "You are a highly knowledgeable NASA mission expert and space historian. "
    "Your role is to provide accurate, detailed answers about NASA space missions "
    "including Apollo 11, Apollo 13, and the Challenger disaster. "
    "Always base your answers on the retrieved source documents provided as context. "
    "When citing information, reference the source document it came from. "
    "If the provided context does not contain enough information to fully answer "
    "a question, clearly state what you can confirm from the sources and indicate "
    "where your knowledge is uncertain or where the context is insufficient. "
    "Do not fabricate facts or make ungrounded claims beyond what the context supports."
)


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Generate a response using OpenAI with retrieved context.

    Args:
        openai_key: OpenAI API key.
        user_message: The user's current question or message.
        context: Formatted context string from retrieved documents.
        conversation_history: List of prior conversation turns
            (each dict has 'role' and 'content').
        model: OpenAI model identifier.
        temperature: Sampling temperature for the model.
        max_tokens: Maximum tokens in the generated response.

    Returns:
        The assistant's response text.
    """
    # Build the messages list starting with the system prompt
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (only role + content, no extraneous keys)
    for turn in conversation_history:
        messages.append({
            "role": turn["role"],
            "content": turn["content"],
        })

    # Build the user message with context if available
    if context and context.strip():
        user_content = (
            f"Use the following retrieved context to answer the question.\n\n"
            f"{context}\n\n"
            f"Question: {user_message}"
        )
    else:
        user_content = user_message

    messages.append({"role": "user", "content": user_content})

    # Create the OpenAI client and send request
    client = OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content
