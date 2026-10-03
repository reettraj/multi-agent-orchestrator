"""Checks for the initial shared state contract."""

from customer_support.state import CustomerSupportState


def test_customer_support_state_declares_messages() -> None:
    assert "messages" in CustomerSupportState.__annotations__
