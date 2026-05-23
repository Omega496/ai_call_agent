"""
Scheduler node.
Handles multi-turn slot filling and invokes the appointment tool to book hospital visits.
"""

import json
import re
import logging
import datetime
from core.state import AgentState
from core.config import settings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from agent.tools.appointment_tool import book_appointment

logger = logging.getLogger(__name__)

VALID_DEPARTMENTS = [
    "cardiology", "radiology", "orthopedics", "general_medicine",
    "pediatrics", "neurology", "oncology", "emergency"
]

def parse_json_from_llm(output: str) -> dict:
    """Helper to parse a JSON object from LLM response safely, stripping markdown wrappers."""
    cleaned = output.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error parsing LLM output: {output}. Error: {e}")
        return {}

async def scheduler_node(state: AgentState) -> AgentState:
    """
    Multi-turn appointment scheduler with slot filling.
    
    Each invocation either:
    a) Asks for the next missing slot (returns question as tts_response)
    b) Fills a slot from the latest transcript
    c) Books the appointment when all slots are filled
    
    State persists between turns via state["appointment_slots"].
    """
    history = state.get("conversation_history", [])
    
    # 1. Get the latest HumanMessage transcript from conversation_history
    latest_human_msg = None
    for msg in reversed(history):
        if isinstance(msg, HumanMessage):
            latest_human_msg = msg
            break
            
    query = latest_human_msg.content.strip() if latest_human_msg else ""
    
    # Check current state of slots
    slots = state.get("appointment_slots", {
        "patient_name": None,
        "department": None,
        "preferred_date": None
    })
    
    # Ensure default keys exist
    for key in ["patient_name", "department", "preferred_date"]:
        if key not in slots:
            slots[key] = None
            
    invalid_department_flag = False
    invalid_department_value = None
    
    # 2. Use LLM to extract slot values from the latest transcript
    if query:
        model = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.0  # Zero temperature for deterministic extraction
        )
        
        today_str = datetime.date.today().isoformat()
        prompt = (
            "Extract appointment booking information from the patient's message.\n"
            "Return a JSON object with keys: patient_name, department, preferred_date.\n"
            "Use null for any value not mentioned.\n\n"
            f"Valid departments: {VALID_DEPARTMENTS}\n"
            f"Reference: Today's date is {today_str}.\n"
            "For dates, convert natural language expressions (e.g. 'tomorrow', 'next Monday') to YYYY-MM-DD format.\n"
            "Return ONLY the JSON object, no other text.\n\n"
            f"Patient's message: '{query}'"
        )
        
        try:
            logger.info("Extracting slots from query...")
            response = await model.ainvoke(prompt)
            extracted = parse_json_from_llm(response.content)
            
            # Update slots based on extracted information
            if extracted:
                if extracted.get("patient_name"):
                    slots["patient_name"] = extracted["patient_name"]
                
                if extracted.get("preferred_date"):
                    slots["preferred_date"] = extracted["preferred_date"]
                
                dept = extracted.get("department")
                if dept:
                    dept_normalized = dept.lower().strip().replace(" ", "_")
                    if dept_normalized in VALID_DEPARTMENTS:
                        slots["department"] = dept_normalized
                    else:
                        invalid_department_flag = True
                        invalid_department_value = dept
                        
        except Exception as e:
            logger.error(f"Error in scheduler extraction: {e}")
            
    # Update slots in state
    state["appointment_slots"] = slots
    
    # 3. Generate natural question asking for the missing slot or book if complete
    if invalid_department_flag:
        # Prompt validation warning
        response_text = (
            f"I'm sorry, we do not have a {invalid_department_value} department. "
            "Our available departments are cardiology, radiology, orthopedics, general medicine, "
            "pediatrics, neurology, oncology, and emergency. Which of these would you prefer?"
        )
    elif not slots["patient_name"]:
        response_text = "To schedule your appointment, could you please tell me your full name?"
    elif not slots["department"]:
        response_text = (
            "Which department would you like to book your appointment with? "
            "We offer cardiology, radiology, orthopedics, general medicine, pediatrics, "
            "neurology, oncology, and emergency."
        )
    elif not slots["preferred_date"]:
        response_text = "What date would you prefer for your appointment? Please let me know your preferred day."
    else:
        # 4. If all slots filled: call book_appointment tool, generate confirmation response
        try:
            logger.info(f"Booking appointment with slots: {slots}")
            booking_res = book_appointment.invoke({
                "patient_name": slots["patient_name"],
                "department": slots["department"],
                "preferred_date": slots["preferred_date"]
            })
            
            if booking_res.get("status") == "confirmed" or "id" in booking_res or "appointment_id" in booking_res:
                conf_num = booking_res.get("confirmation_number", "")
                conf_part = f" Your confirmation number is {conf_num}." if conf_num else ""
                response_text = (
                    f"Thank you. Your appointment has been booked successfully for {slots['patient_name']} "
                    f"with the {slots['department'].replace('_', ' ')} department on {slots['preferred_date']}.{conf_part}"
                )
            else:
                response_text = (
                    "I'm sorry, I encountered an issue booking your appointment. "
                    "Let me connect you with a receptionist who can help."
                )
        except Exception as e:
            logger.error(f"Error booking appointment in scheduler: {e}")
            response_text = (
                "I'm sorry, I'm having trouble connecting to our booking system right now. "
                "Let me connect you with a receptionist who can help."
            )
            
    # 6. Set state["tts_response"]
    state["tts_response"] = response_text
    
    # 7. Append messages to conversation_history
    state["conversation_history"].append(AIMessage(content=response_text))
    
    logger.info(f"Scheduler node finished. Slots: {slots} | TTS Response: '{response_text}'")
    return state
