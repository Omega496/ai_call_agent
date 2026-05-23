"""
State definition for the LangGraph state machine.
Defines the AgentState schema containing call metadata, conversation history, and slot filling states.
"""

from datetime import datetime, timezone
from typing import TypedDict, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    AgentState representing the active caller session and LLM orchestrator state.
    Used for routing, dialogue tracking, slot-filling, and state machine decisions.
    """
    call_sid: str           # Twilio CallSid — for REST API calls (escalation, hangup)
    stream_sid: str         # Twilio StreamSid — for WebSocket audio routing
    caller_number: str
    call_start_time: str    # ISO 8601 timestamp
    conversation_history: list[BaseMessage]
    current_intent: Optional[str]  # answer_faq | check_order_status | schedule_appointment | escalate_to_human
    appointment_slots: dict        # {"patient_name": None, "department": None, "preferred_date": None}
    silence_warnings: int          # 0, 1, or 2 — hangup triggered at 2
    is_speaking: bool              # True while TTS audio is streaming to caller
    tts_response: Optional[str]    # Pending response for the TTS pipeline
    pending_transfer: bool         # True if call transfer to human is pending

def initial_state(call_sid: str, stream_sid: str, caller_number: str) -> AgentState:
    """
    Create a fresh AgentState for a new incoming call.

    Args:
        call_sid: Unique identifier for the Twilio call.
        stream_sid: Unique identifier for the Twilio media stream.
        caller_number: Phone number of the incoming caller.

    Returns:
        An AgentState dictionary initialized with starting values.
    """
    return AgentState(
        call_sid=call_sid,
        stream_sid=stream_sid,
        caller_number=caller_number,
        call_start_time=datetime.now(timezone.utc).isoformat(),
        conversation_history=[],
        current_intent=None,
        appointment_slots={
            "patient_name": None,
            "department": None,
            "preferred_date": None
        },
        silence_warnings=0,
        is_speaking=False,
        tts_response=None,
        pending_transfer=False
    )
