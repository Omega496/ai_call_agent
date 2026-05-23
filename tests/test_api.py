"""
Unit tests for the main FastAPI routes (health check, webhook, etc.).
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """Test that GET /health returns 200 and correct status json."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert json_data["port"] == 5000

def test_twilio_webhook():
    """Test that POST /webhook returns valid XML TwiML connecting the media stream."""
    # Mock Twilio POST request form parameters
    data = {
        "From": "+12345678901",
        "CallSid": "CA12345"
    }
    response = client.post("/webhook", data=data)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    
    # Parse xml content
    xml_content = response.text
    assert "<Response>" in xml_content
    assert "<Connect>" in xml_content
    assert "<Stream" in xml_content
    assert "caller_number=+12345678901" in xml_content
    assert "/twilio" in xml_content

def test_get_order_success():
    """Test that GET /orders/ORD-10001 returns 200 with correct order details."""
    response = client.get("/orders/ORD-10001")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["order_id"] == "ORD-10001"
    assert json_data["order_type"] == "Blood Panel"
    assert json_data["status"] == "completed"

def test_get_order_not_found():
    """Test that GET /orders/ORD-99999 returns 404 with detail and order_id."""
    response = client.get("/orders/ORD-99999")
    assert response.status_code == 404
    json_data = response.json()
    assert json_data["detail"] == "Order not found"
    assert json_data["order_id"] == "ORD-99999"

def test_appointments_lifecycle():
    """Test the full lifecycle of creating, retrieving, listing, and cancelling appointments."""
    # 1. Create appointment
    payload = {
        "patient_name": "Jane Doe",
        "department": "cardiology",
        "preferred_date": "2026-06-20",
        "status": "pending"
    }
    response = client.post("/appointments/", json=payload)
    assert response.status_code == 201
    created_app = response.json()
    assert created_app["patient_name"] == "Jane Doe"
    assert created_app["department"] == "cardiology"
    assert created_app["preferred_date"] == "2026-06-20"
    assert created_app["scheduled_date"] == "2026-06-20"
    assert created_app["status"] == "confirmed"
    assert "appointment_id" in created_app
    assert "confirmation_number" in created_app
    
    app_id = created_app["appointment_id"]
    
    # 2. Get appointment by ID
    response = client.get(f"/appointments/{app_id}")
    assert response.status_code == 200
    retrieved_app = response.json()
    assert retrieved_app["appointment_id"] == app_id
    assert retrieved_app["patient_name"] == "Jane Doe"
    
    # 3. Get appointment not found
    response = client.get("/appointments/APT-NOT-REAL")
    assert response.status_code == 404
    assert response.json()["detail"] == "Appointment not found"
    
    # 4. List appointments
    response = client.get("/appointments/")
    assert response.status_code == 200
    all_apps = response.json()
    assert len(all_apps) > 0
    assert any(a["appointment_id"] == app_id for a in all_apps)
    
    # 5. List with query filter
    response = client.get("/appointments/?patient_name=Jane")
    assert response.status_code == 200
    filtered_apps = response.json()
    assert len(filtered_apps) > 0
    assert all("Jane" in a["patient_name"] for a in filtered_apps)
    
    # 6. Delete (cancel) appointment
    response = client.delete(f"/appointments/{app_id}")
    assert response.status_code == 200
    cancelled_app = response.json()
    assert cancelled_app["appointment_id"] == app_id
    assert cancelled_app["status"] == "cancelled"

def test_metrics_endpoint():
    """Test that GET /metrics returns 200 and returns a list of metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_debug_last_call_endpoint():
    """Test that GET /debug/last-call returns 200 and can be retrieved."""
    import telephony.twilio_handler as handler
    original_val = handler.LAST_CALL_DIAGNOSTICS
    try:
        handler.LAST_CALL_DIAGNOSTICS = {
            "call_sid": "CAtest123",
            "start_event_received": True,
            "audio_chunks_received": 150,
            "speech_final_count": 3,
            "tts_chunks_sent": 50,
            "detected_issues": [],
            "errors": [],
            "connection_duration_sec": 12.5
        }
        response = client.get("/debug/last-call")
        assert response.status_code == 200
        json_data = response.json()
        assert json_data["call_sid"] == "CAtest123"
        assert json_data["audio_chunks_received"] == 150
        assert json_data["speech_final_count"] == 3
        assert json_data["tts_chunks_sent"] == 50
    finally:
        handler.LAST_CALL_DIAGNOSTICS = original_val


