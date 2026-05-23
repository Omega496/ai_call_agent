"""
Classifier node.
Classifies the caller's intent into one of the designated categories using local LLM.
"""

import logging
from core.state import AgentState
from core.config import settings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

# Valid intents defined in the agent specifications
VALID_INTENTS = {
    "answer_faq",
    "check_order_status",
    "schedule_appointment",
    "escalate_to_human"
}

# Initialize ChatOllama model singleton with low temperature for consistent classification
chat_model = ChatOllama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    temperature=0.1
)

async def classify_intent_node(state: AgentState) -> AgentState:
    """
    Classify the latest message in conversation_history into one of 4 intents.
    Updates state["current_intent"] and returns updated state.
    """
    history = state.get("conversation_history", [])
    
    # Extract the latest HumanMessage from state["conversation_history"]
    latest_human_msg = None
    for msg in reversed(history):
        if isinstance(msg, HumanMessage):
            latest_human_msg = msg
            break
            
    if not latest_human_msg:
        logger.warning("No HumanMessage found in conversation history. Defaulting to 'answer_faq'.")
        state["current_intent"] = "answer_faq"
        return state
        
    transcript = latest_human_msg.content.strip()
    
    # Define classification system prompt rules
    system_prompt = (
        "You are an intent classifier for a hospital voice agent.\n"
        "Classify the patient's message into exactly one of these intents:\n"
        "- answer_faq\n"
        "- check_order_status\n"
        "- schedule_appointment\n"
        "- escalate_to_human\n\n"
        "Rules:\n"
        "- Respond with ONLY the intent string, nothing else\n"
        "- If the patient mentions pain, emergency, or distress → escalate_to_human\n"
        "- If the patient asks for a human, operator, or agent → escalate_to_human\n"
        "- If uncertain → answer_faq"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=transcript)
    ]
    
    try:
        # Call local Ollama model
        response = await chat_model.ainvoke(messages)
        intent = response.content.strip().lower()
        
        # Attempt recovery if model returns verbose output or markdown formatting
        if intent not in VALID_INTENTS:
            found_intent = None
            for vi in VALID_INTENTS:
                if vi in intent:
                    found_intent = vi
                    break
            if found_intent:
                intent = found_intent
                
        # Validate against known intents
        if intent in VALID_INTENTS:
            state["current_intent"] = intent
            logger.info(f"Transcript: '{transcript[:60]}...' | Classified Intent: '{intent}'")
        else:
            logger.warning(f"Invalid intent returned by LLM: '{response.content}'. Defaulting to 'answer_faq'.")
            state["current_intent"] = "answer_faq"
            
    except Exception as e:
        logger.error(f"Error calling Ollama classifier model: {e}. Defaulting to 'answer_faq'.")
        state["current_intent"] = "answer_faq"
        
    return state
