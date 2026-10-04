"""Tests for the deterministic LangChain order-access tools."""

from datetime import date

from langchain_core.tools import StructuredTool

from customer_support.operations_tools import get_order_details, get_order_status


def test_get_order_status_returns_processing_for_order_1002() -> None:
    result = get_order_status.invoke({"order_id": "1002"})

    assert result["success"] is True
    assert result["order_id"] == "1002"
    assert result["order_status"] == "Processing"
    assert result["error"] is None


def test_get_order_details_returns_expected_order_1002_data() -> None:
    result = get_order_details.invoke({"order_id": "1002"})

    assert result == {
        "success": True,
        "order_id": "1002",
        "purchase_date": result["purchase_date"],
        "order_status": "Processing",
        "item_name": "Sneakers",
        "is_final_sale": False,
        "is_damaged": False,
        "issue_description": None,
        "error": None,
    }
    date.fromisoformat(result["purchase_date"])


def test_get_order_details_exposes_order_1003_purchase_date() -> None:
    result = get_order_details.invoke({"order_id": "1003"})

    assert result["success"] is True
    assert result["order_id"] == "1003"
    assert result["purchase_date"] is not None
    assert date.fromisoformat(result["purchase_date"])
    assert result["order_status"] == "Delivered"


def test_nonexistent_order_returns_structured_failure() -> None:
    status_result = get_order_status.invoke({"order_id": "999999"})
    details_result = get_order_details.invoke({"order_id": "999999"})

    assert status_result == {
        "success": False,
        "order_id": "999999",
        "order_status": None,
        "error": "order_not_found",
    }
    assert details_result["success"] is False
    assert details_result["purchase_date"] is None
    assert details_result["order_status"] is None
    assert details_result["error"] == "order_not_found"


def test_invalid_order_id_returns_structured_failure() -> None:
    result = get_order_details.invoke({"order_id": "not-an-order-id"})

    assert result["success"] is False
    assert result["order_id"] == "not-an-order-id"
    assert result["error"] == "invalid_order_id"
    assert result["purchase_date"] is None


def test_tools_expose_langchain_tool_calling_interface() -> None:
    assert isinstance(get_order_status, StructuredTool)
    assert isinstance(get_order_details, StructuredTool)
    assert get_order_status.name == "get_order_status"
    assert get_order_details.name == "get_order_details"
    assert set(get_order_status.args) == {"order_id"}
    assert set(get_order_details.args) == {"order_id"}
    assert "status of an order" in get_order_status.description.lower()
    assert "order" in get_order_details.description.lower()
