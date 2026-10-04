"""Grounded customer-facing response generation from specialist results."""

from customer_support.agents.supervisor import Route
from customer_support.state import CustomerSupportState


MISSING_RESULT_RESPONSE = (
    "I don't have enough information in the current assessment to provide an update."
)


def generate_final_response(state: CustomerSupportState) -> dict[str, str]:
    """Render a concise response using only the selected agent's structured result.

    This deterministic renderer does not make policy, order, or escalation
    decisions. It reuses the selected specialist's result fields so it cannot add
    facts beyond those already present in shared state.
    """

    route: Route | None = state.get("route")  # type: ignore[typeddict-item]
    if route == "policy":
        result = state.get("policy_result")
        response = _text_or_fallback(result.get("summary") if result else None)
    elif route == "operations":
        result = state.get("operations_result")
        response = _operations_response(result)
    elif route == "escalation":
        result = state.get("escalation_result")
        response = _escalation_response(result)
    else:
        response = MISSING_RESULT_RESPONSE

    return {"final_response": response}


def _operations_response(result: dict[str, object] | None) -> str:
    if result is None:
        return MISSING_RESULT_RESPONSE

    outcome = result.get("outcome")
    if outcome == "order_status_retrieved" and result.get("order_status"):
        return f"The current status of your order is {result['order_status']}."

    if outcome == "order_details_retrieved":
        details: list[str] = []
        if result.get("order_status"):
            details.append(f"The order status is {result['order_status']}.")
        if result.get("purchase_date"):
            details.append(f"The purchase date was {result['purchase_date']}.")
        if details:
            return " ".join(details)

    return _text_or_fallback(result.get("summary"))


def _escalation_response(result: dict[str, object] | None) -> str:
    if result is None:
        return MISSING_RESULT_RESPONSE

    reason = str(result.get("reason") or "").strip()
    if result.get("required") is True:
        if not reason:
            return "This request requires human review."
        return f"Human review is required: {reason}"
    return _text_or_fallback(reason)


def _text_or_fallback(value: object) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return MISSING_RESULT_RESPONSE
