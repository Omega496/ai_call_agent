"""
Order lookup tool.
Queries the mock API for order status by order ID.
"""

import logging
import requests
from langchain_core.tools import tool
from core.config import settings

logger = logging.getLogger(__name__)

@tool
def check_order_status(order_id: str) -> dict:
    """
    Check the status of a hospital order by order ID.
    
    Use this tool when a patient asks about the status of their lab order,
    test result, prescription, or any other hospital order.
    
    Args:
        order_id: The hospital order identifier. Usually starts with 'ORD-'
                  followed by digits (e.g., 'ORD-12345'). Extract this from
                  the patient's spoken message.
    
    Returns:
        dict with keys: order_id, order_type, status, department,
                        ordered_date, expected_date, notes
        On error: dict with keys: error, order_id
    """
    url = f"{settings.mock_api_base_url}/orders/{order_id}"
    logger.debug(f"Calling GET {url} with timeout 5.0")
    try:
        response = requests.get(url, timeout=5.0)
        if response.status_code == 404:
            logger.warning(f"Order {order_id} not found (404)")
            return {"error": "Order not found", "order_id": order_id}
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        logger.error(f"Timeout checking order {order_id}")
        return {"error": "Service temporarily unavailable", "order_id": order_id}
    except requests.ConnectionError:
        logger.error(f"Connection error checking order {order_id}")
        return {"error": "Connection failed", "order_id": order_id}
    except Exception as e:
        logger.error(f"Error checking order {order_id}: {e}")
        return {"error": str(e), "order_id": order_id}

@tool
def query_order_status(order_id: str) -> str:
    """Check the status of a specific hospital order by ID."""
    res = check_order_status.invoke({"order_id": order_id})
    return str(res)
