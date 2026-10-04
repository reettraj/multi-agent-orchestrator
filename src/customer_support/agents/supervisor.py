"""Front-door routing decisions for the customer support agents."""

import os
import re
from collections.abc import Mapping
from typing import Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from customer_support.state import CustomerSupportState


Route = Literal["policy", "operations", "escalation"]
DEFAULT_CHAT_MODEL = "gpt-4o-mini"

SUPERVISOR_SYSTEM_PROMPT = """You are the routing Supervisor for a customer-support system.
Choose exactly one specialist route: policy, operations, or escalation. Do not
answer the customer, draft a response, or perform the specialist's work.

Use policy for questions about store rules such as returns, sale-item eligibility,
shipping, sizing, fit, exchanges, and refund policy information. Use operations
for transactional lookups such as an order's current status, purchase date, or
other order details. Use escalation for transactional refund requests, strong
anger or frustration, damaged/defective orders, or requests for human approval
or an exception. An informational question about the refund policy belongs to
policy; a request to issue a refund belongs to escalation. For an eligibility
question about a specific order, use operations first so its facts can be checked
before policy assessment. Base the route and brief reason only on the supplied
request and shared-state context."""


class SupervisorDecision(TypedDict):
    """Routing output for a future graph edge selector."""

    route: Route
    reason: str


class _RoutingAssessment(BaseModel):
    """Validated structured model response for semantically unclear requests."""

    route: Route = Field(description="Specialist that should handle the request")
    reason: str = Field(description="Brief internal explanation of the route")


def supervisor_agent(
    state: CustomerSupportState,
    *,
    llm: BaseChatModel | None = None,
) -> SupervisorDecision:
    """Select a specialist route without producing a customer-facing answer.

    Clear policy, order-lookup, and escalation requests are classified with
    deterministic rules. Structured LLM classification is reserved for requests
    that do not match those rules. The LLM is injectable for tests.
    """

    request_text = (state.get("request_text") or "").strip()
    deterministic = _deterministic_route(state, request_text)
    if deterministic is not None:
        return deterministic

    if not request_text:
        return {"route": "escalation", "reason": "The request is empty or unavailable for classification."}

    try:
        model = (llm or _create_chat_model()).with_structured_output(_RoutingAssessment)
        assessment = _RoutingAssessment.model_validate(
            model.invoke(_build_messages(state, request_text))
        )
        return {
            "route": assessment.route,
            "reason": assessment.reason.strip() or "Selected by structured request classification.",
        }
    except Exception:
        # An uncertain route should be reviewed rather than silently dropped.
        return {
            "route": "escalation",
            "reason": "The request could not be classified reliably and needs review.",
        }


def _create_chat_model() -> ChatOpenAI:
    """Create the configured OpenAI chat model."""

    return ChatOpenAI(
        model=os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        temperature=0,
    )


def _deterministic_route(
    state: Mapping[str, object], request_text: str
) -> SupervisorDecision | None:
    """Route requests with explicit, high-confidence intent signals."""

    text = request_text.lower()
    intent = _normalize_intent(state.get("intent"))
    existing_escalation = state.get("escalation_result")
    if isinstance(existing_escalation, Mapping) and existing_escalation.get("required") is True:
        reason = str(existing_escalation.get("reason") or "An escalation assessment requires human review.")
        return {"route": "escalation", "reason": reason}

    if _is_escalation_request(text, state, intent):
        return {
            "route": "escalation",
            "reason": "The request includes a refund, strong negative sentiment, an item issue, or a need for human authorization.",
        }

    if requires_order_policy_lookup(state, request_text):
        return {
            "route": "operations",
            "reason": "Order details are needed before evaluating the return or refund policy.",
        }

    # Ordinary policy requests remain single-agent policy routes.
    if _is_policy_request(text, intent):
        return {"route": "policy", "reason": "The request asks about a store policy or product-fit guidance."}

    order_id = state.get("order_id")
    has_order_reference = bool(order_id) or bool(
        re.search(r"#\s*\d+\b|\border\b.{0,20}\b\d+\b", text)
    )
    if intent in {"order_status", "order_details", "order_lookup", "operations"} or (
        has_order_reference and _is_order_lookup(text)
    ):
        return {"route": "operations", "reason": "The request asks for transactional information about a specific order."}

    return None


def requires_order_policy_lookup(
    state: Mapping[str, object], request_text: str | None = None
) -> bool:
    """Return whether an order-specific return/refund policy needs order facts."""

    text = (
        request_text
        if request_text is not None
        else str(state.get("request_text") or "")
    ).lower()
    intent = _normalize_intent(state.get("intent"))
    has_order_reference = bool(state.get("order_id")) or bool(
        re.search(r"#\s*\d+\b|\border\b.{0,20}\b\d+\b", text)
    )
    policy_question = intent in {
        "return_eligibility",
        "return_request",
        "refund_eligibility",
        "exchange_eligibility",
    } or bool(
        re.search(r"\b(return(?:ing)?|exchange)\b", text)
        or (
            re.search(r"\brefund\b", text)
            and re.search(r"\b(eligib(?:le|ility)|qualif(?:y|ies)|allowed)\b", text)
        )
    )
    return has_order_reference and policy_question


def _is_escalation_request(text: str, state: Mapping[str, object], intent: str) -> bool:
    """Recognize clear risk and human-review requests, excluding refund FAQs."""

    if intent in {"refund", "refund_request", "request_refund", "escalation"}:
        return True

    sentiment = str(state.get("customer_sentiment") or "").lower()
    if re.search(r"\b(angry|furious|frustrated|infuriated|enraged)\b", text) or any(
        word in sentiment for word in ("angry", "furious", "frustrat", "infuriat", "enrag")
    ):
        return True

    if re.search(
        r"\b(damaged|defective|broken|unsafe|dangerous|malfunction(?:ing)?|not working)\b",
        text,
    ):
        return True

    if re.search(
        r"\b(manager|supervisor|authorization|authorisation|approval|approve|override|exception)\b|"
        r"\b(?:speak|talk|connect|transfer)\s+(?:me\s+)?to\s+(?:a\s+)?human(?:\s+agent)?\b",
        text,
    ):
        return True

    transactional_refund = re.search(
        r"\b(want|need|request|issue|process|approve|give|get|send|receive)\b"
        r".{0,35}\b(?:a\s+)?refund\b|\brefund\s+(?:me|my)\b|"
        r"\b(?:want|need|request|get|give)\b.{0,30}\b(?:my\s+)?money\s+back\b",
        text,
    )
    return bool(transactional_refund)


def _is_policy_request(text: str, intent: str) -> bool:
    if intent in {"policy", "return_policy", "shipping", "sizing", "size", "fit", "exchange"}:
        return True
    return bool(
        re.search(
            r"\b(return(?:s|ing)?|final\s+sale|sale\s+item|shipping|ship(?:ping)?\s+policy|"
            r"size(?:\s+chart|ing)?|fit|exchange(?:s|d)?)\b",
            text,
        )
        or re.search(r"\brefund\b.{0,30}\b(policy|timing|method|when|how long)\b", text)
    )


def _is_order_lookup(text: str) -> bool:
    return bool(
        re.search(
            r"\b(status|where.{0,50}\bis\b|where's|track(?:ing)?|purchase\s+date|when\s+.*\bordered|"
            r"order\s+details|details\s+for|look\s+up)\b",
            text,
        )
    )


def _normalize_intent(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _build_messages(state: CustomerSupportState, request_text: str) -> list[SystemMessage | HumanMessage]:
    context = {
        "intent": state.get("intent"),
        "customer_sentiment": state.get("customer_sentiment"),
        "order_id": state.get("order_id"),
        "escalation_result": state.get("escalation_result"),
    }
    return [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Customer request:\n{request_text}\n\nShared-state context:\n{context}"),
    ]
