"""Escalation decision agent, separate from any human-review workflow."""

import json
import os
import re
from collections.abc import Mapping
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from customer_support.state import CustomerSupportState, EscalationResult


DEFAULT_CHAT_MODEL = "gpt-4o-mini"

ESCALATION_SYSTEM_PROMPT = """You decide whether a customer-support request needs
human escalation. Consider the request, stated customer sentiment, and existing
policy/operations results. Escalate requests for a refund or human authorization,
reports of damaged or seriously defective orders, and customers expressing
strong anger or frustration. Do not automatically escalate ordinary order-status
questions or informational policy questions. Do not initiate human review or
interrupt a workflow; only return the escalation decision.

When escalation is required, provide a specific reason and a priority: low,
medium, high, or urgent. Use urgent only for immediate safety concerns; otherwise
choose a priority that reflects the customer's situation. When escalation is
not required, provide a brief reason and set priority to null. Base the decision
only on information in the request and supplied state."""


class _EscalationAssessment(BaseModel):
    """Structured response validated before conversion to shared state."""

    required: bool = Field(description="Whether the request needs human escalation")
    reason: str = Field(description="Short reason supporting the decision")
    priority: Literal["low", "medium", "high", "urgent"] | None = Field(
        description="Escalation priority, or null when escalation is not required"
    )


def escalation_agent(
    state: CustomerSupportState,
    *,
    llm: BaseChatModel | None = None,
) -> dict[str, EscalationResult]:
    """Assess whether the request needs a human escalation.

    The LLM returns a structured decision. Clear high-risk triggers in the
    request/state are applied as a safety override so a model false negative
    cannot suppress a refund, serious damage, anger, or authorization escalation.
    This function only records a decision; it does not start human review.
    """

    request_text = (state.get("request_text") or "").strip()
    if not request_text:
        return {
            "escalation_result": {
                "required": False,
                "reason": "No customer request was provided to assess.",
                "priority": None,
            }
        }

    signals = _detect_escalation_signals(state, request_text)
    model = llm or _create_chat_model()
    try:
        structured_model = model.with_structured_output(_EscalationAssessment)
        assessment = _EscalationAssessment.model_validate(
            structured_model.invoke(_build_messages(state, request_text))
        )
    except Exception:
        if signals:
            return {"escalation_result": _result_from_signals(signals)}
        return {
            "escalation_result": {
                "required": True,
                "reason": (
                    "The automated escalation assessment could not be completed; "
                    "human review is needed."
                ),
                "priority": "high",
            }
        }

    if signals:
        return {"escalation_result": _result_from_signals(signals)}

    if assessment.required:
        return {
            "escalation_result": {
                "required": True,
                "reason": assessment.reason.strip()
                or "The request needs human review.",
                "priority": assessment.priority or "medium",
            }
        }

    return {
        "escalation_result": {
            "required": False,
            "reason": assessment.reason.strip()
            or "No escalation trigger was identified.",
            "priority": None,
        }
    }


def _create_chat_model() -> ChatOpenAI:
    """Create the configured OpenAI model used for escalation assessment."""

    return ChatOpenAI(
        model=os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        temperature=0,
    )


def _build_messages(
    state: CustomerSupportState,
    request_text: str,
) -> list[SystemMessage | HumanMessage]:
    """Include the request and relevant upstream results in the model context."""

    context = {
        "customer_sentiment": state.get("customer_sentiment"),
        "policy_result": state.get("policy_result"),
        "operations_result": state.get("operations_result"),
    }
    return [
        SystemMessage(content=ESCALATION_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Customer request:\n{request_text}\n\n"
                f"Current support context:\n{json.dumps(context, ensure_ascii=False, default=str)}"
            )
        ),
    ]


def _detect_escalation_signals(
    state: Mapping[str, object],
    request_text: str,
) -> list[tuple[str, str, str]]:
    """Return explicit escalation signals with reason and priority values."""

    text = request_text.lower()
    signals: list[tuple[str, str, str]] = []

    if _refund_requested(text, state.get("intent")):
        signals.append(
            (
                "refund",
                "The customer is requesting a refund, which needs human review.",
                "medium",
            )
        )

    sentiment = str(state.get("customer_sentiment") or "").lower()
    if re.search(r"\b(angry|furious|frustrated|infuriated|enraged)\b", text) or any(
        label in sentiment
        for label in ("angry", "furious", "frustrat", "infuriat", "enrag")
    ):
        signals.append(
            (
                "emotion",
                "The customer is expressing strong anger or frustration and needs human follow-up.",
                "high",
            )
        )

    if re.search(
        r"\b(damaged|defective|broken|unsafe|dangerous|malfunction(?:ing)?|not working)\b",
        text,
    ):
        priority = "urgent" if re.search(r"\b(unsafe|dangerous)\b", text) else "high"
        signals.append(
            (
                "item_issue",
                "The customer reports a damaged or defective item that needs human assessment.",
                priority,
            )
        )

    if re.search(
        r"\b(manager|supervisor|human(?:\s+agent)?|person|authorization|authorisation|"
        r"approval|approve|override|exception)\b",
        text,
    ):
        signals.append(
            (
                "authorization",
                "The request requires human authorization or an exception review.",
                "high",
            )
        )

    return signals


def _refund_requested(text: str, intent: object) -> bool:
    """Detect transactional refund requests without flagging refund FAQs."""

    normalized_intent = str(intent or "").lower().replace("-", "_").replace(" ", "_")
    if normalized_intent in {"refund", "refund_request", "request_refund"}:
        return True

    return bool(
        re.search(
            r"\b(want|need|request|issue|process|approve|give|get|send|receive)\b"
            r".{0,35}\b(?:a\s+)?refund\b",
            text,
        )
        or re.search(r"\brefund\s+(?:me|my|for\s+order)\b", text)
        or re.search(
            r"\b(?:want|need|request|get|give)\b.{0,30}\b(?:my\s+)?money\s+back\b",
            text,
        )
        or re.search(r"\b(?:get|give)\s+my\s+money\s+back\b", text)
    )


def _result_from_signals(
    signals: list[tuple[str, str, str]],
) -> EscalationResult:
    """Build a reasoned escalation result from explicit risk signals."""

    priority_rank = {"low": 0, "medium": 1, "high": 2, "urgent": 3}
    priority = max((signal[2] for signal in signals), key=priority_rank.__getitem__)
    reason = " ".join(signal[1] for signal in signals)
    return {"required": True, "reason": reason, "priority": priority}
