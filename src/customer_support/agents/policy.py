"""Policy assessment agent using retrieved store-policy context."""

import json
import os
from collections.abc import Callable, Sequence
from datetime import date
from typing import Literal

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from customer_support.policy_retrieval import search_policy_chunks
from customer_support.state import CustomerSupportState, OperationsResult, PolicyResult


DEFAULT_CHAT_MODEL = "gpt-4o-mini"

POLICY_SYSTEM_PROMPT = """You are the Policy Agent for a customer-support team.
Assess the customer's request using only the supplied retrieved policy context.
Ground every policy claim in that context. Never invent or assume a store rule.
If the context does not answer a point, use null for the corresponding factual
field and say what is unknown in the summary. Separate policy rules from facts
about the customer's particular item or order.

Set final_sale to true only when the request or supplied facts identify the item
as sale/final-sale, false only when it is established as not sale/final-sale,
and null when that fact is unknown. Set return_eligible only when both the
relevant policy and the facts in the request/context support a conclusion;
otherwise use null. When system-retrieved order context includes a purchase date
and elapsed calendar days, compare that age against the return window stated in
the retrieved policy. Do not assume a window if the policy context does not
state one. return_window_days should report the window stated in the retrieved
policy, or null if no window is stated. An exception for a reported damaged,
defective, or incorrectly fulfilled item means review is needed; do not promise
approval.

Use one of these decision values: eligible, not_eligible, review_required,
insufficient_information. Return only the requested structured fields."""


class _PolicyAssessment(BaseModel):
    """Validated structured response, converted to the shared state contract."""

    decision: Literal[
        "eligible", "not_eligible", "review_required", "insufficient_information"
    ] = Field(description="Policy outcome for this request")
    summary: str = Field(description="Brief explanation grounded in retrieved text")
    final_sale: bool | None = Field(
        description="Whether the item is identified as final sale, or null if unknown"
    )
    return_window_days: int | None = Field(
        description="Return window in calendar days, or null if not stated"
    )
    return_eligible: bool | None = Field(
        description="Whether the specific return is eligible, or null if unknown"
    )


PolicyRetriever = Callable[[str], Sequence[Document]]


def policy_agent(
    state: CustomerSupportState,
    *,
    llm: BaseChatModel | None = None,
    retriever: PolicyRetriever | None = None,
) -> dict[str, PolicyResult]:
    """Assess a request against retrieved policy and return a state update.

    ``llm`` and ``retriever`` can be injected for testing or alternate runtime
    configuration. With no LLM supplied, an OpenAI chat model is configured
    from ``OPENAI_CHAT_MODEL`` (default ``gpt-4o-mini``) and ``OPENAI_API_KEY``.
    """

    request_text = (state.get("request_text") or "").strip()
    if not request_text:
        return {"policy_result": _insufficient_context_result()}

    retrieve = retriever or search_policy_chunks
    documents = retrieve(request_text)
    context = _format_policy_context(documents)
    if not context:
        return {"policy_result": _insufficient_context_result()}

    model = llm or _create_chat_model()
    structured_model = model.with_structured_output(_PolicyAssessment)
    order_context = _format_order_context(state.get("operations_result"))
    messages = [
        SystemMessage(content=POLICY_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Customer request:\n{request_text}\n\n"
                f"Retrieved policy context:\n{context}\n\n"
                f"Verified order context:\n{order_context}"
            )
        ),
    ]
    assessment = _PolicyAssessment.model_validate(structured_model.invoke(messages))
    result: PolicyResult = assessment.model_dump()
    return {"policy_result": result}


def _create_chat_model() -> ChatOpenAI:
    """Create the configured OpenAI chat model without loading credentials here."""

    return ChatOpenAI(
        model=os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        temperature=0,
    )


def _format_policy_context(documents: Sequence[Document]) -> str:
    """Format retrieved text with its available section metadata."""

    formatted: list[str] = []
    for document in documents:
        text = document.page_content.strip()
        if not text:
            continue

        metadata = document.metadata
        headings = [
            metadata[key]
            for key in ("section", "subsection")
            if metadata.get(key)
        ]
        source = " / ".join(headings) or metadata.get("source", "Policy")
        formatted.append(f"[{source}]\n{text}")
    return "\n\n".join(formatted)


def _format_order_context(result: OperationsResult | None) -> str:
    """Provide verified order facts and elapsed purchase age to policy reasoning."""

    if not result:
        return "No order details have been retrieved."

    facts: dict[str, str | int | None] = {
        "order_status": result.get("order_status"),
        "purchase_date": result.get("purchase_date"),
        "days_since_purchase": None,
    }
    purchase_date = result.get("purchase_date")
    if purchase_date:
        try:
            facts["days_since_purchase"] = (
                date.today() - date.fromisoformat(purchase_date)
            ).days
        except ValueError:
            # Preserve the source date but do not infer an age from malformed data.
            pass
    return json.dumps(facts, ensure_ascii=False, sort_keys=True)


def _insufficient_context_result() -> PolicyResult:
    """Return a safe result when no applicable policy text was retrieved."""

    return {
        "decision": "insufficient_information",
        "summary": (
            "I could not find relevant policy information to assess this request. "
            "No policy eligibility decision can be made from the available context."
        ),
        "final_sale": None,
        "return_window_days": None,
        "return_eligible": None,
    }
