"""Tests for SQLite order persistence and sample-data initialization."""

import sqlite3
from datetime import date

from customer_support.order_database import get_order, initialize_database


def test_database_and_orders_table_are_created(tmp_path) -> None:
    database_path = tmp_path / "orders.db"

    initialized_path = initialize_database(database_path)

    assert initialized_path == database_path
    assert database_path.is_file()
    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(orders)")
        }

    assert {
        "order_id",
        "purchase_date",
        "order_status",
        "item_name",
        "is_final_sale",
        "is_damaged",
        "issue_description",
    } <= columns


def test_initialization_seeds_required_scenario_orders(tmp_path) -> None:
    database_path = initialize_database(tmp_path / "orders.db")

    order_1001 = get_order("1001", database_path)
    order_1002 = get_order("1002", database_path)
    order_1003 = get_order("1003", database_path)

    assert order_1001 is not None
    assert order_1001["item_name"] == "Hoodie"
    assert order_1001["is_final_sale"] is True
    assert order_1002 is not None
    assert order_1002["order_status"] == "Processing"
    assert order_1003 is not None
    assert order_1003["order_status"] == "Delivered"


def test_order_1002_can_be_retrieved(tmp_path) -> None:
    database_path = initialize_database(tmp_path / "orders.db")

    order = get_order("1002", database_path)

    assert order == {
        "order_id": "1002",
        "purchase_date": order["purchase_date"],
        "order_status": "Processing",
        "item_name": "Sneakers",
        "is_final_sale": False,
        "is_damaged": False,
        "issue_description": None,
    }


def test_order_1003_is_about_40_days_old(tmp_path) -> None:
    database_path = initialize_database(tmp_path / "orders.db")

    order = get_order("1003", database_path)

    assert order is not None
    days_since_purchase = (date.today() - date.fromisoformat(order["purchase_date"])).days
    assert days_since_purchase == 40
    assert days_since_purchase > 30


def test_missing_order_returns_none(tmp_path) -> None:
    database_path = initialize_database(tmp_path / "orders.db")

    assert get_order("9999", database_path) is None


def test_initialization_is_idempotent_and_preserves_existing_rows(tmp_path) -> None:
    database_path = tmp_path / "orders.db"
    initialize_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE orders SET order_status = ? WHERE order_id = ?",
            ("Manual Review", "1002"),
        )

    initialize_database(database_path)

    assert get_order("1002", database_path)["order_status"] == "Manual Review"
    with sqlite3.connect(database_path) as connection:
        order_count = connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    assert order_count == 3
