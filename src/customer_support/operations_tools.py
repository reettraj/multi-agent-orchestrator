"""Deterministic LangChain tools for reading order records."""

from typing import TypedDict

from langchain_core.tools import tool

from customer_support.order_database import OrderRecord, get_order


class OrderStatusToolResult(TypedDict):
    """Stable response contract for an order-status lookup."""

    success: bool
    order_id: str
    order_status: str | None
    error: str | None


class OrderDetailsToolResult(TypedDict):
    """Stable response contract for an order-details lookup."""

    success: bool
    order_id: str
    purchase_date: str | None
    order_status: str | None
    item_name: str | None
    is_final_sale: bool | None
    is_damaged: bool | None
    issue_description: str | None
    error: str | None


def _find_order(
    order_id: str,
) -> tuple[str, OrderRecord | None, str | None]:
    """Validate a tool argument and perform a database-layer lookup."""

    normalized_id = order_id.strip()
    if not normalized_id or not normalized_id.isdecimal():
        return normalized_id, None, "invalid_order_id"

    order = get_order(normalized_id)
    if order is None:
        return normalized_id, None, "order_not_found"
    return normalized_id, order, None


@tool
def get_order_status(order_id: str) -> OrderStatusToolResult:
    """Get the current status of an order by its numeric order ID."""

    normalized_id, order, error = _find_order(order_id)
    return {
        "success": order is not None,
        "order_id": normalized_id,
        "order_status": order["order_status"] if order is not None else None,
        "error": error,
    }


@tool
def get_order_details(order_id: str) -> OrderDetailsToolResult:
    """Get dates, status, item, and issue-related details for an order ID."""

    normalized_id, order, error = _find_order(order_id)
    if order is None:
        return {
            "success": False,
            "order_id": normalized_id,
            "purchase_date": None,
            "order_status": None,
            "item_name": None,
            "is_final_sale": None,
            "is_damaged": None,
            "issue_description": None,
            "error": error,
        }

    return {
        "success": True,
        "order_id": order["order_id"],
        "purchase_date": order["purchase_date"],
        "order_status": order["order_status"],
        "item_name": order["item_name"],
        "is_final_sale": order["is_final_sale"],
        "is_damaged": order["is_damaged"],
        "issue_description": order["issue_description"],
        "error": None,
    }
