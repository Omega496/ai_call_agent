"""
Escalation node.
Triggers a Twilio Dial redirect to transfer the caller to a human agent.
"""

import logging
from core.state import AgentState
from core.config import settings
from twilio.rest import Client
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

def transfer_call_to_human(call_sid: str) -> bool:
    """
    Redirect live call to human agent using Twilio REST API.
    Returns True on success, False on failure.
    """
    try:
        logger.info(f"Initiating Twilio redirect for Call SID: {call_sid} to {settings.human_agent_number}")
        client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
        client.calls(call_sid).update(
            twiml=f'<Response><Dial>{settings.human_agent_number}</Dial></Response>'
        )
        return True
    except Exception as e:
        logger.error(f"Failed to transfer call {call_sid}: {e}")
        return False

async def escalation_node(state: AgentState) -> AgentState:
    """
    Transfer the live Twilio call to a human agent.
    
    1. Generate empathetic transfer message
    2. Set tts_response so the message plays BEFORE transfer
    3. Set a flag indicating transfer should happen after TTS completes
    """
    history = state.get("conversation_history", [])
    
    # Check context for urgent words in recent conversation history
    is_urgent = False
    urgent_keywords = ["pain", "emergency", "urgent", "chest", "bleeding", "severe", "hurt"]
    
    for msg in history:
        content_lower = msg.content.lower()
        if any(keyword in content_lower for keyword in urgent_keywords):
            is_urgent = True
            break
            
    if is_urgent:
        response_text = "I understand this is urgent. I'm connecting you with our care team right away. Please hold."
    else:
        response_text = "I'm connecting you with one of our care team members right away. Please hold."
        
    state["tts_response"] = response_text
    state["conversation_history"].append(AIMessage(content=response_text))
    state["pending_transfer"] = True
    
    logger.info(f"Escalation node finished. Urgent: {is_urgent}. Response: '{response_text}'")
    return state
