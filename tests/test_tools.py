import pytest
from unittest.mock import patch, MagicMock
import requests
from agent.tools.order_tool import check_order_status
from agent.tools.appointment_tool import book_appointment

# 1. test_check_order_status_found
def test_check_order_status_found():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "order_id": "ORD-10001",
        "order_type": "Blood Panel",
        "status": "completed",
        "department": "Laboratory"
    }
    
    with patch("requests.get", return_value=mock_response) as mock_get:
        res = check_order_status.invoke({"order_id": "ORD-10001"})
        assert res["order_id"] == "ORD-10001"
        assert res["order_type"] == "Blood Panel"
        assert res["status"] == "completed"
        assert res["department"] == "Laboratory"
        mock_get.assert_called_once()

# 2. test_check_order_status_not_found
def test_check_order_status_not_found():
    mock_response = MagicMock()
    mock_response.status_code = 404
    
    with patch("requests.get", return_value=mock_response) as mock_get:
        res = check_order_status.invoke({"order_id": "ORD-99999"})
        assert res == {"error": "Order not found", "order_id": "ORD-99999"}
        mock_get.assert_called_once()

# 3. test_check_order_status_timeout
def test_check_order_status_timeout():
    with patch("requests.get", side_effect=requests.Timeout) as mock_get:
        res = check_order_status.invoke({"order_id": "ORD-123"})
        assert "error" in res
        assert res["error"] == "Service temporarily unavailable"
        mock_get.assert_called_once()

# 4. test_book_appointment_success
def test_book_appointment_success():
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {
        "appointment_id": "APT-XYZ789",
        "patient_name": "Jane Doe",
        "department": "cardiology",
        "preferred_date": "2026-06-15",
        "status": "confirmed",
        "confirmation_number": "CONF-123456"
    }
    
    with patch("requests.post", return_value=mock_response) as mock_post:
        res = book_appointment.invoke({
            "patient_name": "Jane Doe",
            "department": "cardiology",
            "preferred_date": "2026-06-15"
        })
        assert res["appointment_id"] == "APT-XYZ789"
        assert res["confirmation_number"] == "CONF-123456"
        mock_post.assert_called_once()

# 5. test_book_appointment_api_error
def test_book_appointment_api_error():
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = requests.HTTPError("Internal Server Error")
    
    with patch("requests.post", return_value=mock_response) as mock_post:
        res = book_appointment.invoke({
            "patient_name": "Jane Doe",
            "department": "cardiology",
            "preferred_date": "2026-06-15"
        })
        assert "error" in res
        assert res["error"] == "Unexpected error"
        mock_post.assert_called_once()

# 6. test_book_appointment_connection_error
def test_book_appointment_connection_error():
    with patch("requests.post", side_effect=requests.ConnectionError) as mock_post:
        res = book_appointment.invoke({
            "patient_name": "Jane Doe",
            "department": "cardiology",
            "preferred_date": "2026-06-15"
        })
        assert "error" in res
        assert res["error"] == "Connection failed"
        mock_post.assert_called_once()

# 7. test_tool_docstrings_are_descriptive
def test_tool_docstrings_are_descriptive():
    # Assert check_order_status docstring
    doc_order = check_order_status.func.__doc__
    assert doc_order is not None
    assert len(doc_order) > 100
    assert "order_id" in doc_order
    
    # Assert book_appointment docstring
    doc_book = book_appointment.func.__doc__
    assert doc_book is not None
    assert len(doc_book) > 100
    assert "patient_name" in doc_book
    assert "department" in doc_book
    assert "preferred_date" in doc_book
