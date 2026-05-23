"""
FAQ node.
Retrieves context using RAG and answers patient questions using the local LLM.
"""

import logging
from typing import Optional
from core.state import AgentState
from core.config import settings
from rag.retriever import retrieve_chunks
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

# Initialize ChatOllama model with temperature 0.3 for FAQ responses
chat_model = ChatOllama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    temperature=0.3
)

async def faq_node(state: AgentState) -> AgentState:
    """
    RAG-powered FAQ answering node.
    
    1. Extracts latest HumanMessage as the query
    2. Retrieves top-5 chunks from ChromaDB
    3. Builds prompt with retrieved context + conversation history
    4. Streams LLM response
    5. Appends AIMessage to conversation_history
    6. Returns updated state with response stored for TTS
    """
    history = state.get("conversation_history", [])
    
    # 1. Get the latest HumanMessage transcript from conversation_history
    latest_human_msg = None
    for msg in reversed(history):
        if isinstance(msg, HumanMessage):
            latest_human_msg = msg
            break
            
    query = latest_human_msg.content.strip() if latest_human_msg else ""
    
    # 2. Call retrieve_chunks(query, n_results=5)
    chunks = []
    try:
        if query:
            chunks = retrieve_chunks(query, n_results=5)
        logger.info(f"Query: '{query}' | Retrieved {len(chunks)} chunks.")
    except Exception as e:
        logger.error(f"Error retrieving FAQ context: {e}")
        chunks = []

    # 3. Format chunks as numbered list in the prompt
    retrieved_chunks_str = ""
    for idx, chunk in enumerate(chunks, 1):
        retrieved_chunks_str += f"{idx}. {chunk}\n"
    if not chunks:
        retrieved_chunks_str = "(No context found)"

    # Format conversation history
    history_str = ""
    for msg in history:
        role = "Patient" if isinstance(msg, HumanMessage) else "Assistant"
        history_str += f"{role}: {msg.content}\n"

    # Build system prompt from template
    system_prompt = (
        "You are a helpful hospital receptionist answering a patient's question over the phone.\n\n"
        "Guidelines:\n"
        "- Keep responses under 3 sentences — this will be spoken aloud\n"
        "- Speak naturally as if on a phone call, not like a written document\n"
        "- Never mention \"according to our records\" or \"based on the information provided\"\n"
        "- If the retrieved context does not contain the answer, say:\n"
        "  \"I don't have that information right now. Would you like me to connect you with someone who can help?\"\n"
        "- Never hallucinate medical information. If unsure, offer to escalate.\n\n"
        "RETRIEVED CONTEXT:\n"
        f"{retrieved_chunks_str}\n\n"
        "CONVERSATION HISTORY:\n"
        f"{history_str}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please provide the receptionist's spoken response to the patient.")
    ]

    try:
        # 4. Call ChatOllama with stream=True (via astream)
        response_chunks = []
        async for chunk in chat_model.astream(messages):
            response_chunks.append(chunk.content)
            
        # 5. Accumulate streamed tokens into full response string
        response = "".join(response_chunks).strip()
        
        # Graceful fallback if empty response
        if not response:
            response = "I don't have that information right now. Would you like me to connect you with someone who can help?"
            
    except Exception as e:
        logger.error(f"Error calling ChatOllama in FAQ node: {e}")
        response = "I don't have that information right now. Would you like me to connect you with someone who can help?"

    # 6. Append AIMessage(content=response) to state["conversation_history"]
    state["conversation_history"].append(AIMessage(content=response))
    
    # 7. Store response in state for the TTS pipeline to consume
    state["tts_response"] = response
    
    logger.info(f"Generated FAQ response (length {len(response)} chars): '{response[:60]}...'")
    return state
