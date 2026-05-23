"""
Mock orders router.
Provides endpoint GET /orders/{order_id} to query medical/prescription order statuses.
"""

from typing import Optional
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()

class OrderResponse(BaseModel):
    order_id: str
    order_type: str
    status: str
    department: str
    ordered_date: str
    expected_date: str
    completed_date: Optional[str] = None
    notes: Optional[str] = None

class OrderErrorResponse(BaseModel):
    detail: str
    order_id: str

MOCK_ORDERS = {
    "ORD-10001": {
        "order_id": "ORD-10001",
        "order_type": "Blood Panel",
        "status": "completed",
        "department": "Laboratory",
        "ordered_date": "2026-05-18",
        "expected_date": "2026-05-20",
        "completed_date": "2026-05-19",
        "notes": "Results available in patient portal"
    },
    "ORD-10002": {
        "order_id": "ORD-10002",
        "order_type": "MRI Scan",
        "status": "scheduled",
        "department": "Radiology",
        "ordered_date": "2026-05-20",
        "expected_date": "2026-05-28",
        "completed_date": None,
        "notes": "Appointment confirmed for 2026-05-28 at 10:30 AM"
    },
    "ORD-10003": {
        "order_id": "ORD-10003",
        "order_type": "Prescription - Amoxicillin",
        "status": "ready_for_pickup",
        "department": "Pharmacy",
        "ordered_date": "2026-05-22",
        "expected_date": "2026-05-23",
        "completed_date": None,
        "notes": "Ready at main pharmacy counter"
    },
    "ORD-10004": {
        "order_id": "ORD-10004",
        "order_type": "X-Ray",
        "status": "processing",
        "department": "Radiology",
        "ordered_date": "2026-05-23",
        "expected_date": "2026-05-24",
        "completed_date": None,
        "notes": "Being reviewed by radiologist"
    },
    "ORD-10005": {
        "order_id": "ORD-10005",
        "order_type": "Cardiology Consultation",
        "status": "pending",
        "department": "Cardiology",
        "ordered_date": "2026-05-23",
        "expected_date": "2026-06-05",
        "completed_date": None,
        "notes": "Awaiting specialist availability"
    }
}

@router.get(
    "/{order_id}",
    response_model=OrderResponse,
    responses={404: {"model": OrderErrorResponse}}
)
def get_order(order_id: str):
    """
    Retrieve details for a specific hospital order.
    """
    order = MOCK_ORDERS.get(order_id)
    if not order:
        return JSONResponse(
            status_code=404,
            content={"detail": "Order not found", "order_id": order_id}
        )
    return order
