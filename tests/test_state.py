"""Checks for the shared customer support state contract."""

from operator import add
from typing import Annotated, Literal, get_args, get_origin, get_type_hints

from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages

from customer_support.state import (
    AgentError,
    CustomerSupportState,
    EscalationResult,
    HumanDecision,
    OperationsResult,
    PolicyResult,
)


def test_customer_support_state_declares_all_fields() -> None:
    expected_fields = {
        "messages",
        "request_id",
        "request_text",
        "order_id",
        "intent",
        "customer_sentiment",
        "policy_result",
        "operations_result",
        "escalation_result",
        "human_decision",
        "final_response",
        "errors",
    }

    assert set(CustomerSupportState.__annotations__) == expected_fields
    assert CustomerSupportState.__required_keys__ == expected_fields


def test_optional_state_fields_are_explicitly_nullable() -> None:
    hints = get_type_hints(CustomerSupportState, include_extras=True)
    optional_fields = {
        "order_id",
        "intent",
        "customer_sentiment",
        "policy_result",
        "operations_result",
        "escalation_result",
        "human_decision",
        "final_response",
    }

    for field in optional_fields:
        assert type(None) in get_args(hints[field])


def test_messages_use_langgraph_add_messages_reducer() -> None:
    messages_hint = get_type_hints(CustomerSupportState, include_extras=True)["messages"]

    assert get_origin(messages_hint) is Annotated
    message_type, reducer = get_args(messages_hint)
    assert message_type.__origin__ is list
    assert reducer is add_messages


def test_errors_use_additive_list_reducer() -> None:
    errors_hint = get_type_hints(CustomerSupportState, include_extras=True)["errors"]

    assert get_origin(errors_hint) is Annotated
    error_list_type, reducer = get_args(errors_hint)
    assert error_list_type.__origin__ is list
    assert reducer is add

    first: AgentError = {
        "agent": "policy",
        "code": "lookup_failed",
        "message": "Policy lookup failed.",
        "retryable": True,
    }
    second: AgentError = {
        "agent": "operations",
        "code": "order_not_found",
        "message": "Order was not found.",
        "retryable": False,
    }
    assert reducer([first], [second]) == [first, second]


def test_message_reducer_appends_distinct_messages() -> None:
    reducer = get_args(
        get_type_hints(CustomerSupportState, include_extras=True)["messages"]
    )[1]
    previous = [HumanMessage(content="First message")]
    update = [HumanMessage(content="Second message")]

    result = reducer(previous, update)

    assert [message.content for message in result] == [
        "First message",
        "Second message",
    ]


def test_agent_result_structures_have_expected_fields() -> None:
    assert set(PolicyResult.__annotations__) == {
        "decision",
        "summary",
        "final_sale",
        "return_window_days",
        "return_eligible",
    }
    assert set(OperationsResult.__annotations__) == {
        "outcome",
        "summary",
        "order_status",
        "purchase_date",
    }
    assert set(EscalationResult.__annotations__) == {
        "required",
        "reason",
        "priority",
    }
    assert set(HumanDecision.__annotations__) == {"decision", "reason"}
    assert get_type_hints(HumanDecision)["decision"] == Literal["approved", "rejected"]
    assert set(AgentError.__annotations__) == {
        "agent",
        "code",
        "message",
        "retryable",
    }
