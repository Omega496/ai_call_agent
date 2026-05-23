"""
Order status node.
Retrieves status information for hospital client orders using tool calling.
"""

import logging
from core.state import AgentState
from core.config import settings
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from agent.tools.order_tool import check_order_status

logger = logging.getLogger(__name__)

# System prompt directing the receptionist's task
SYSTEM_PROMPT = (
    "You are a hospital receptionist checking a patient's order status over the phone.\n"
    "The patient has provided information about their order. Extract the order ID and\n"
    "check its status using the available tool.\n\n"
    "If no order ID is mentioned, ask the patient: \"Could you please provide your order ID? \n"
    "It should be on your paperwork and starts with ORD followed by numbers.\"\n\n"
    "Keep all responses under 3 sentences. Speak naturally for a phone call.\n"
    "After checking, summarize the order status clearly for the patient."
)

async def order_status_node(state: AgentState) -> AgentState:
    """
    Check hospital order status using LLM tool calling.
    
    Flow:
    1. Bind check_order_status tool to LLM
    2. LLM extracts order_id from transcript and calls tool
    3. Tool result is incorporated into natural language response
    4. Response stored as tts_response and appended to conversation_history
    """
    history = state.get("conversation_history", [])
    
    # Initialize the ChatOllama model with temperature 0.2
    model = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.2
    )
    model_with_tools = model.bind_tools([check_order_status])
    
    # Compile messages including the system prompt and conversation history
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history
    
    try:
        logger.info("Calling LLM to extract order status query details...")
        response = await model_with_tools.ainvoke(messages)
        
        # Check if model returned a tool call
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            logger.info(f"LLM triggered tool '{tool_name}' with args: {tool_args}")
            
            # Execute the tool
            if tool_name == "check_order_status":
                try:
                    tool_result = check_order_status.invoke(tool_args)
                except Exception as ex:
                    logger.error(f"Error executing check_order_status tool: {ex}")
                    tool_result = {"error": f"Tool execution failed: {str(ex)}"}
            else:
                tool_result = {"error": f"Unknown tool called: {tool_name}"}
                
            logger.info(f"Tool execution result: {tool_result}")
            
            # Append intermediate messages to keep the context consistent
            state["conversation_history"].append(response)
            
            tool_message = ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_id,
                name=tool_name
            )
            state["conversation_history"].append(tool_message)
            
            # Invoke LLM a second time with the tool results included
            updated_messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["conversation_history"]
            final_response = await model.ainvoke(updated_messages)
            final_text = final_response.content.strip()
            
            state["conversation_history"].append(AIMessage(content=final_text))
            state["tts_response"] = final_text
            
        else:
            # Model responded directly without needing tools (e.g. requesting order ID)
            final_text = response.content.strip()
            state["conversation_history"].append(AIMessage(content=final_text))
            state["tts_response"] = final_text
            
    except Exception as e:
        logger.error(f"Error in order_status_node loop: {e}")
        fallback_text = (
            "I'm sorry, I'm having trouble looking up your order status right now. "
            "Could you please repeat your order ID, or try again in a few minutes?"
        )
        state["conversation_history"].append(AIMessage(content=fallback_text))
        state["tts_response"] = fallback_text
        
    logger.info(f"Order status node complete. TTS Response: '{state['tts_response']}'")
    return state
