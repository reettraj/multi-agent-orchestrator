"""SQLite setup and read access for the sample order records."""

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "orders.db"


class OrderRecord(TypedDict):
    """Order data used by the Operations Agent and later workflow stages."""

    order_id: str
    purchase_date: str
    order_status: str
    item_name: str
    is_final_sale: bool
    is_damaged: bool
    issue_description: str | None


def initialize_database(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> Path:
    """Create the database, table, and deterministic scenario records.

    Order 1003 is seeded 40 days before the initialization date so it is outside
    the store's 30-day return window. Existing records are left untouched.
    """

    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seed_date = date.today()

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                purchase_date TEXT NOT NULL,
                order_status TEXT NOT NULL,
                item_name TEXT NOT NULL,
                is_final_sale INTEGER NOT NULL DEFAULT 0
                    CHECK (is_final_sale IN (0, 1)),
                is_damaged INTEGER NOT NULL DEFAULT 0
                    CHECK (is_damaged IN (0, 1)),
                issue_description TEXT
            )
            """
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO orders (
                order_id,
                purchase_date,
                order_status,
                item_name,
                is_final_sale,
                is_damaged,
                issue_description
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            _seed_orders(seed_date),
        )

    return path


def get_order(
    order_id: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> OrderRecord | None:
    """Return an order by ID, or ``None`` if no matching order exists."""

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT
                order_id,
                purchase_date,
                order_status,
                item_name,
                is_final_sale,
                is_damaged,
                issue_description
            FROM orders
            WHERE order_id = ?
            """,
            (str(order_id),),
        ).fetchone()

    if row is None:
        return None

    return {
        "order_id": row["order_id"],
        "purchase_date": row["purchase_date"],
        "order_status": row["order_status"],
        "item_name": row["item_name"],
        "is_final_sale": bool(row["is_final_sale"]),
        "is_damaged": bool(row["is_damaged"]),
        "issue_description": row["issue_description"],
    }


def _seed_orders(seed_date: date) -> tuple[tuple[object, ...], ...]:
    """Build stable scenario data relative to the database's first setup date."""

    return (
        (
            "1001",
            (seed_date - timedelta(days=12)).isoformat(),
            "Delivered",
            "Hoodie",
            1,
            0,
            None,
        ),
        (
            "1002",
            (seed_date - timedelta(days=2)).isoformat(),
            "Processing",
            "Sneakers",
            0,
            0,
            None,
        ),
        (
            "1003",
            (seed_date - timedelta(days=40)).isoformat(),
            "Delivered",
            "Jacket",
            0,
            0,
            None,
        ),
    )


if __name__ == "__main__":
    print(f"Initialized order database at {initialize_database()}")
