import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from core.state import AgentState, initial_state

# 1. test_classifier_faq_intent
@pytest.mark.asyncio
async def test_classifier_faq_intent(base_state):
    from agent.nodes.classifier import classify_intent_node
    with patch("langchain_ollama.ChatOllama.ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = AIMessage(content="answer_faq")
        res_state = await classify_intent_node(base_state)
        assert res_state["current_intent"] == "answer_faq"

# 2. test_classifier_order_intent
@pytest.mark.asyncio
async def test_classifier_order_intent(base_state):
    from agent.nodes.classifier import classify_intent_node
    base_state["conversation_history"] = [
        HumanMessage(content="I want to check order ORD-12345")
    ]
    with patch("langchain_ollama.ChatOllama.ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = AIMessage(content="check_order_status")
        res_state = await classify_intent_node(base_state)
        assert res_state["current_intent"] == "check_order_status"

# 3. test_classifier_escalate_intent
@pytest.mark.asyncio
async def test_classifier_escalate_intent(base_state):
    from agent.nodes.classifier import classify_intent_node
    base_state["conversation_history"] = [
        HumanMessage(content="I need help right now, I'm in pain")
    ]
    with patch("langchain_ollama.ChatOllama.ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = AIMessage(content="escalate_to_human")
        res_state = await classify_intent_node(base_state)
        assert res_state["current_intent"] == "escalate_to_human"

# 4. test_classifier_invalid_response_defaults_to_faq
@pytest.mark.asyncio
async def test_classifier_invalid_response_defaults_to_faq(base_state):
    from agent.nodes.classifier import classify_intent_node
    with patch("langchain_ollama.ChatOllama.ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = AIMessage(content="invalid_intent_xyz")
        res_state = await classify_intent_node(base_state)
        assert res_state["current_intent"] == "answer_faq"

# 5. test_faq_node_returns_response
@pytest.mark.asyncio
async def test_faq_node_returns_response(base_state):
    from agent.nodes.faq import faq_node
    
    async def mock_astream(*args, **kwargs):
        yield MagicMock(content="Visiting hours are 9 AM to 8 PM daily.")
        
    with patch("agent.nodes.faq.retrieve_chunks", return_value=["Visiting hours are 9am-8pm"]) as mock_retrieve, \
         patch("langchain_ollama.ChatOllama.astream", side_effect=mock_astream):
        res_state = await faq_node(base_state)
        assert res_state["tts_response"] == "Visiting hours are 9 AM to 8 PM daily."
        assert len(res_state["conversation_history"]) == 2

# 6. test_order_status_node_with_valid_order
@pytest.mark.asyncio
async def test_order_status_node_with_valid_order(base_state):
    from agent.nodes.order_status import order_status_node
    base_state["conversation_history"] = [
        HumanMessage(content="Check order ORD-10001")
    ]
    
    first_resp = AIMessage(content="")
    first_resp.tool_calls = [{
        "name": "check_order_status",
        "args": {"order_id": "ORD-10001"},
        "id": "call_ord1"
    }]
    second_resp = AIMessage(content="Your order is completed.")
    
    async def mock_invoke(*args, **kwargs):
        messages = args[1] if len(args) > 1 else args[0]
        if not isinstance(messages, list):
            messages = kwargs.get("input") or []
        from langchain_core.messages import ToolMessage
        if any(isinstance(m, ToolMessage) for m in messages):
            return second_resp
        return first_resp
        
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "order_id": "ORD-10001",
        "status": "completed",
        "order_type": "Blood Panel"
    }
    
    with patch("langchain_ollama.ChatOllama.ainvoke", side_effect=mock_invoke), \
         patch("requests.get", return_value=mock_response) as mock_get:
        res_state = await order_status_node(base_state)
        assert "completed" in res_state["tts_response"]
        mock_get.assert_called_once()

# 7. test_order_status_node_with_missing_order_id
@pytest.mark.asyncio
async def test_order_status_node_with_missing_order_id(base_state):
    from agent.nodes.order_status import order_status_node
    base_state["conversation_history"] = [
        HumanMessage(content="I want to check my order")
    ]
    with patch("langchain_ollama.ChatOllama.ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = AIMessage(content="Could you please provide your order ID?")
        res_state = await order_status_node(base_state)
        assert "order ID" in res_state["tts_response"]

# 8. test_scheduler_node_collects_first_slot
@pytest.mark.asyncio
async def test_scheduler_node_collects_first_slot(base_state):
    from agent.nodes.scheduler import scheduler_node
    base_state["conversation_history"] = [
        HumanMessage(content="I need to schedule an appointment")
    ]
    base_state["appointment_slots"] = {"patient_name": None, "department": None, "preferred_date": None}
    
    with patch("langchain_ollama.ChatOllama.ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = AIMessage(content='{"patient_name": null, "department": null, "preferred_date": null}')
        res_state = await scheduler_node(base_state)
        assert "name" in res_state["tts_response"].lower()
        assert res_state["appointment_slots"] == {"patient_name": None, "department": None, "preferred_date": None}

# 9. test_scheduler_node_fills_slot_and_asks_next
@pytest.mark.asyncio
async def test_scheduler_node_fills_slot_and_asks_next(base_state):
    from agent.nodes.scheduler import scheduler_node
    base_state["conversation_history"] = [
        HumanMessage(content="My name is John Smith")
    ]
    base_state["appointment_slots"] = {"patient_name": None, "department": None, "preferred_date": None}
    
    with patch("langchain_ollama.ChatOllama.ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = AIMessage(content='{"patient_name": "John Smith", "department": null, "preferred_date": null}')
        res_state = await scheduler_node(base_state)
        assert res_state["appointment_slots"]["patient_name"] == "John Smith"
        assert "department" in res_state["tts_response"].lower()

# 10. test_scheduler_node_books_when_all_slots_filled
@pytest.mark.asyncio
async def test_scheduler_node_books_when_all_slots_filled(base_state):
    from agent.nodes.scheduler import scheduler_node
    base_state["conversation_history"] = [
        HumanMessage(content="My name is John Smith"),
        HumanMessage(content="Cardiology"),
        HumanMessage(content="2026-05-24")
    ]
    base_state["appointment_slots"] = {
        "patient_name": "John Smith",
        "department": "cardiology",
        "preferred_date": "2026-05-24"
    }
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "confirmed",
        "confirmation_number": "CONF-987654"
    }
    
    with patch("langchain_ollama.ChatOllama.ainvoke", new_callable=AsyncMock) as mock_invoke, \
         patch("requests.post", return_value=mock_response):
        mock_invoke.return_value = AIMessage(content='{"patient_name": "John Smith", "department": "cardiology", "preferred_date": "2026-05-24"}')
        res_state = await scheduler_node(base_state)
        assert "CONF-987654" in res_state["tts_response"]

# 11. test_escalation_node_sets_pending_transfer
@pytest.mark.asyncio
async def test_escalation_node_sets_pending_transfer(base_state):
    from agent.nodes.escalation import escalation_node
    res_state = await escalation_node(base_state)
    assert res_state["pending_transfer"] is True
    assert "transfer" in res_state["tts_response"].lower() or "connecting" in res_state["tts_response"].lower()

# 12. test_route_intent_routing
def test_route_intent_routing(base_state):
    from agent.graph import route_intent
    
    base_state["current_intent"] = "answer_faq"
    assert route_intent(base_state) == "faq"
    
    base_state["current_intent"] = "check_order_status"
    assert route_intent(base_state) == "order_status"
    
    base_state["current_intent"] = "schedule_appointment"
    assert route_intent(base_state) == "scheduler"
    
    base_state["current_intent"] = "escalate_to_human"
    assert route_intent(base_state) == "escalation"
