import pytest
from langchain_core.messages import HumanMessage
from core.state import AgentState, initial_state

@pytest.fixture
def base_state() -> AgentState:
    state = initial_state("CA123", "SM456", "+1234567890")
    state["conversation_history"] = [
        HumanMessage(content="What are your visiting hours?")
    ]
    return state
