"""Tests for standalone Supervisor routing decisions."""

from unittest.mock import Mock

from customer_support.agents.supervisor import SupervisorDecision, supervisor_agent
from customer_support.state import CustomerSupportState


def _state(request_text: str, **updates: object) -> CustomerSupportState:
    state: CustomerSupportState = {
        "messages": [],
        "request_id": "request-test",
        "request_text": request_text,
        "order_id": None,
        "intent": None,
        "customer_sentiment": None,
        "policy_result": None,
        "operations_result": None,
        "escalation_result": None,
        "human_decision": None,
        "final_response": None,
        "errors": [],
    }
    state.update(updates)  # type: ignore[arg-type]
    return state


def _structured_llm(route: str, reason: str = "semantic test route") -> Mock:
    structured = Mock()
    structured.invoke.return_value = {"route": route, "reason": reason}
    llm = Mock()
    llm.with_structured_output.return_value = structured
    llm.structured = structured
    return llm


def test_sale_item_return_question_routes_to_policy_without_llm() -> None:
    result = supervisor_agent(_state("Can I return this discounted hoodie?"), llm=Mock())

    assert result["route"] == "policy"
    assert "policy" in result["reason"]


def test_shipping_question_routes_to_policy() -> None:
    result = supervisor_agent(_state("How long does shipping take?"), llm=Mock())

    assert result["route"] == "policy"


def test_order_1002_status_question_routes_to_operations() -> None:
    result = supervisor_agent(
        _state("Can you tell me where my order #1002 is? I ordered it yesterday."), llm=Mock()
    )

    assert result["route"] == "operations"


def test_order_details_purchase_date_routes_to_operations() -> None:
    result = supervisor_agent(_state("What was the purchase date for order #1003?"), llm=Mock())

    assert result["route"] == "operations"


def test_order_specific_return_eligibility_routes_to_operations_first() -> None:
    result = supervisor_agent(
        _state("I received order #1003 about 40 days ago. Can I send it back for a refund?"), llm=Mock()
    )

    assert result["route"] == "operations"


def test_actual_immediate_refund_request_still_routes_to_escalation() -> None:
    result = supervisor_agent(
        _state("I want a refund right now for order #1003."), llm=Mock()
    )

    assert result["route"] == "escalation"


def test_order_reference_alone_does_not_turn_refund_policy_faq_into_lookup() -> None:
    result = supervisor_agent(
        _state("What is the refund policy for order #1003?"), llm=Mock()
    )

    assert result["route"] == "policy"


def test_damaged_order_and_refund_routes_to_escalation() -> None:
    result = supervisor_agent(
        _state("My order #1002 arrived damaged and I want a refund."), llm=Mock()
    )

    assert result["route"] == "escalation"


def test_torn_and_ruined_refund_request_routes_to_escalation() -> None:
    result = supervisor_agent(
        _state("My order #1001 arrived completely torn and ruined! I want a refund right now!"),
        llm=Mock(),
    )

    assert result["route"] == "escalation"


def test_angry_refund_request_routes_to_escalation() -> None:
    result = supervisor_agent(
        _state("I'm furious. I want a refund!", customer_sentiment="angry"), llm=Mock()
    )

    assert result["route"] == "escalation"


def test_supervisor_does_not_generate_a_final_customer_answer() -> None:
    result = supervisor_agent(_state("Where is order #1002?"), llm=Mock())

    assert set(result) == {"route", "reason"}
    assert "final_response" not in result
    assert "messages" not in result


def test_routing_decision_has_structured_format() -> None:
    result = supervisor_agent(_state("How do I exchange a wrong size?"), llm=Mock())

    assert isinstance(result, dict)
    assert set(result) == {"route", "reason"}
    assert result["route"] in {"policy", "operations", "escalation"}
    assert isinstance(result["reason"], str) and result["reason"]
    assert SupervisorDecision.__required_keys__ == {"route", "reason"}


def test_ambiguous_request_uses_structured_llm_classification() -> None:
    llm = _structured_llm("operations", "This request concerns an account lookup.")

    result = supervisor_agent(_state("I need help with my account."), llm=llm)

    assert result == {"route": "operations", "reason": "This request concerns an account lookup."}
    llm.with_structured_output.assert_called_once()
    llm.structured.invoke.assert_called_once()


def test_existing_required_escalation_result_takes_precedence() -> None:
    result = supervisor_agent(
        _state(
            "Where is order #1002?",
            escalation_result={"required": True, "reason": "Already flagged for review.", "priority": "high"},
        ),
        llm=Mock(),
    )

    assert result == {"route": "escalation", "reason": "Already flagged for review."}
