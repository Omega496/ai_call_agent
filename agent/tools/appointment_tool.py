"""
Appointment booking tool.
Invokes the SQLite backend API to schedule and store hospital appointments.
"""

import logging
import requests
from langchain_core.tools import tool
from core.config import settings

logger = logging.getLogger(__name__)

@tool
def book_appointment(patient_name: str, department: str, preferred_date: str) -> dict:
    """
    Book a hospital appointment for a patient.
    
    ONLY call this tool when you have collected ALL THREE required pieces of
    information from the patient. Do not call with null or placeholder values.
    
    Args:
        patient_name: Full name of the patient as spoken (e.g., "John Smith")
        department: Hospital department. Must be exactly one of:
                   cardiology, radiology, orthopedics, general_medicine,
                   pediatrics, neurology, oncology, emergency
        preferred_date: Date in YYYY-MM-DD format (e.g., "2026-06-15")
    
    Returns:
        dict with keys: appointment_id, patient_name, department,
                        scheduled_date, status, confirmation_number
        On error: dict with keys: error, message
    """
    url = f"{settings.mock_api_base_url}/appointments"
    logger.debug(f"Calling POST {url} with body json, timeout 5.0")
    try:
        response = requests.post(
            url,
            json={
                "patient_name": patient_name,
                "department": department,
                "preferred_date": preferred_date,
                "status": "pending"
            },
            timeout=5.0
        )
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        logger.error("Timeout booking appointment")
        return {"error": "Service temporarily unavailable", "message": "The booking service timed out. Please try again."}
    except requests.ConnectionError:
        logger.error("Connection error booking appointment")
        return {"error": "Connection failed", "message": "Failed to connect to the booking service. Please try again."}
    except Exception as e:
        logger.error(f"Error booking appointment: {e}")
        return {"error": "Unexpected error", "message": str(e)}
