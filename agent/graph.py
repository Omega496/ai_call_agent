"""
LangGraph StateGraph definition.
Orchestrates the routing between classifying intents, FAQ answers, order status, scheduling, and escalation.
"""

import logging
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph as CompiledGraph
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import tracing_context

from core.state import AgentState
from core.config import settings
from core.latency import LatencyTracker
from agent.nodes.classifier import classify_intent_node
from agent.nodes.faq import faq_node
from agent.nodes.order_status import order_status_node
from agent.nodes.scheduler import scheduler_node
from agent.nodes.escalation import escalation_node

logger = logging.getLogger(__name__)

def route_intent(state: AgentState) -> str:
    """Read current_intent from state and return the target node name."""
    intent = state.get("current_intent", "answer_faq")
    valid_intents = {
        "answer_faq": "faq",
        "check_order_status": "order_status",
        "schedule_appointment": "scheduler",
        "escalate_to_human": "escalation",
    }
    return valid_intents.get(intent, "faq")  # default to faq if unknown

def build_graph() -> CompiledGraph:
    """
    Build and compile the LangGraph StateGraph state machine.
    """
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("faq", faq_node)
    graph.add_node("order_status", order_status_node)
    graph.add_node("scheduler", scheduler_node)
    graph.add_node("escalation", escalation_node)
    
    # Set entry point
    graph.set_entry_point("classify_intent")
    
    # Add conditional routing edge
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "faq": "faq",
            "order_status": "order_status",
            "scheduler": "scheduler",
            "escalation": "escalation",
        }
    )
    
    # Connect leaf nodes to END
    graph.add_edge("faq", END)
    graph.add_edge("order_status", END)
    graph.add_edge("scheduler", END)
    graph.add_edge("escalation", END)
    
    return graph.compile()

# Module-level compiled graph singleton
agent_graph = build_graph()

async def run_agent_turn(state: AgentState, transcript: str) -> tuple[AgentState, str]:
    """
    Run one turn of the agent pipeline.
    
    Args:
        state: Current AgentState
        transcript: New transcript from Deepgram speech_final event
        
    Returns:
        (updated_state, response_text) where response_text is the TTS string
    """
    # Append new transcript as a HumanMessage
    state["conversation_history"].append(HumanMessage(content=transcript))
    
    # Run the compiled graph wrapped with LangSmith tracing context and custom metadata
    config = {
        "metadata": {
            "call_sid": state.get("call_sid"),
            "caller_number": state.get("caller_number"),
            "intent": state.get("current_intent"),
        },
        "tags": ["voice-agent", "hospital", "production"],
        "run_name": f"turn_{len(state['conversation_history'])}_{state.get('current_intent', 'unknown')}"
    }
    call_sid = state.get("call_sid", "unknown")
    with LatencyTracker("graph_invocation", call_sid=call_sid) as tracker:
        with tracing_context(project_name=settings.langchain_project):
            updated_state = await agent_graph.ainvoke(state, config=config)
            
    # Manually check and log warning if the threshold is exceeded for the classified intent
    intent = updated_state.get("current_intent", "unknown")
    is_rag = (intent == "answer_faq")
    graph_threshold = 3000.0 if is_rag else 2000.0
    if tracker.elapsed_ms > graph_threshold:
        logger.warning(
            f"[{call_sid}] graph_invocation: {tracker.elapsed_ms:.0f}ms (exceeded {graph_threshold}ms, intent: {intent})"
        )
        
    # Extract the response text
    response_text = updated_state.get("tts_response")
    
    if not response_text:
        history = updated_state.get("conversation_history", [])
        if history and isinstance(history[-1], AIMessage):
            response_text = history[-1].content
        else:
            # Fallback if no response is generated (e.g. stubs)
            response_text = f"I received your message: {transcript}. How can I assist you with your health inquiry today?"
            history.append(AIMessage(content=response_text))
            
    # Keep tts_response synchronized
    updated_state["tts_response"] = response_text
        
    return updated_state, response_text
