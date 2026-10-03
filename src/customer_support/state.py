"""Shared state contract for the customer support workflow."""

from operator import add
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class PolicyResult(TypedDict):
    """Policy assessment shared with the other workflow nodes."""

    decision: str
    summary: str
    final_sale: bool | None
    return_window_days: int | None
    return_eligible: bool | None


class OperationsResult(TypedDict):
    """Outcome and relevant order details from an operations task."""

    outcome: str
    summary: str
    order_status: str | None
    purchase_date: str | None


class EscalationResult(TypedDict):
    """Assessment of whether a request needs escalation."""

    required: bool
    reason: str
    priority: str | None


class HumanDecision(TypedDict):
    """A human approval or rejection result recorded in shared state."""

    decision: Literal["approved", "rejected"]
    reason: str | None


class AgentError(TypedDict):
    """Structured error information reported by a workflow agent."""

    agent: str
    code: str
    message: str
    retryable: bool


class CustomerSupportState(TypedDict):
    """Shared request data and results produced during a support workflow."""

    messages: Annotated[list[AnyMessage], add_messages]
    request_id: str
    request_text: str
    order_id: str | None
    intent: str | None
    customer_sentiment: str | None
    policy_result: PolicyResult | None
    operations_result: OperationsResult | None
    escalation_result: EscalationResult | None
    human_decision: HumanDecision | None
    final_response: str | None
    errors: Annotated[list[AgentError], add]
