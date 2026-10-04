"""Tests for Escalation Agent decisions without external model calls."""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from customer_support.agents.escalation import escalation_agent
from customer_support.state import EscalationResult


class FakeStructuredRunnable:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.messages: Sequence[BaseMessage] | None = None

    def invoke(self, messages: Sequence[BaseMessage]) -> dict[str, object]:
        self.messages = messages
        return self.response


class FakeChatModel:
    def __init__(self, response: dict[str, object]) -> None:
        self.structured = FakeStructuredRunnable(response)
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured


def _assessment(
    required: bool,
    reason: str,
    priority: str | None,
) -> dict[str, object]:
    return {"required": required, "reason": reason, "priority": priority}


def _text_from_messages(messages: Sequence[BaseMessage] | None) -> str:
    assert messages is not None
    return "\n".join(str(message.content) for message in messages)


def test_damaged_order_and_refund_request_requires_escalation() -> None:
    model = FakeChatModel(_assessment(False, "No escalation needed.", None))

    result = escalation_agent(
        {
            "request_text": "My order arrived damaged and I want a refund.",
            "customer_sentiment": "concerned",
            "operations_result": {
                "outcome": "order_details_retrieved",
                "summary": "The delivered jacket was damaged.",
                "order_status": "Delivered",
                "purchase_date": "2026-09-01",
            },
        },
        llm=model,
    )["escalation_result"]

    assert result["required"] is True
    assert "refund" in result["reason"].lower()
    assert "damaged" in result["reason"].lower()


def test_torn_and_ruined_order_is_recognized_as_damaged() -> None:
    result = escalation_agent(
        {"request_text": "My order #1001 arrived completely torn and ruined! I want a refund right now!"},
        llm=FakeChatModel(_assessment(False, "No issue found.", None)),
    )["escalation_result"]

    assert result["required"] is True
    assert "damaged or defective" in result["reason"].lower()
    assert "refund" in result["reason"].lower()
    assert result["priority"] == "high"


def test_angry_customer_requesting_refund_requires_escalation() -> None:
    model = FakeChatModel(_assessment(False, "No escalation needed.", None))

    result = escalation_agent(
        {
            "request_text": "I'm furious. I want my money back now.",
            "customer_sentiment": "angry",
        },
        llm=model,
    )["escalation_result"]

    assert result["required"] is True
    assert result["priority"] == "high"
    assert "refund" in result["reason"].lower() or "money back" in result["reason"].lower()
    assert "anger" in result["reason"].lower()


def test_normal_order_status_question_does_not_escalate() -> None:
    model = FakeChatModel(
        _assessment(False, "This is a routine order-status question.", "low")
    )

    result = escalation_agent(
        {"request_text": "Where is order #1002?", "intent": "order_status"},
        llm=model,
    )["escalation_result"]

    assert result == {
        "required": False,
        "reason": "This is a routine order-status question.",
        "priority": None,
    }


def test_ordinary_policy_question_does_not_escalate() -> None:
    model = FakeChatModel(
        _assessment(False, "The customer is asking for policy information only.", None)
    )

    result = escalation_agent(
        {"request_text": "What is the store's refund timing policy?"},
        llm=model,
    )["escalation_result"]

    assert result["required"] is False
    assert result["priority"] is None


def test_required_result_has_reason_and_priority() -> None:
    model = FakeChatModel(_assessment(True, "A manager exception was requested.", None))

    result = escalation_agent(
        {"request_text": "Could a manager approve an exception to this policy?"},
        llm=model,
    )["escalation_result"]

    assert result["required"] is True
    assert result["reason"]
    assert result["priority"] == "high"


def test_result_matches_existing_escalation_result_structure() -> None:
    model = FakeChatModel(
        _assessment(False, "No human escalation trigger was identified.", None)
    )

    result: EscalationResult = escalation_agent(
        {"request_text": "How do I find the size chart?"}, llm=model
    )["escalation_result"]

    assert set(result) == {"required", "reason", "priority"}
    assert isinstance(result["required"], bool)
    assert isinstance(result["reason"], str)
    assert result["priority"] is None
    assert model.schema is not None


def test_structured_llm_receives_current_request_and_support_context() -> None:
    model = FakeChatModel(_assessment(False, "No escalation trigger found.", None))

    escalation_agent(
        {
            "request_text": "The item arrived with a cracked handle.",
            "customer_sentiment": "upset",
            "policy_result": {
                "decision": "review_required",
                "summary": "Damaged items can be reported for review.",
                "final_sale": False,
                "return_window_days": 30,
                "return_eligible": None,
            },
            "operations_result": {
                "outcome": "order_details_retrieved",
                "summary": "Order 1003 details retrieved.",
                "order_status": "Delivered",
                "purchase_date": "2026-08-25",
            },
        },
        llm=model,
    )

    prompt_text = _text_from_messages(model.structured.messages)
    assert "cracked handle" in prompt_text
    assert "upset" in prompt_text
    assert "review_required" in prompt_text
    assert "2026-08-25" in prompt_text
