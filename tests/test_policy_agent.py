"""Focused tests for the Policy Agent without external API calls."""

from collections.abc import Sequence
from datetime import date, timedelta

from langchain_core.documents import Document

from customer_support.agents.policy import policy_agent
from customer_support.state import PolicyResult


class FakeStructuredModel:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.messages = None
        self.invocation_count = 0

    def invoke(self, messages):
        self.messages = messages
        self.invocation_count += 1
        return self.response


class FakeChatModel:
    def __init__(self, response: dict[str, object]) -> None:
        self.structured = FakeStructuredModel(response)
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured


def _response(
    *,
    decision: str,
    summary: str,
    final_sale: bool | None,
    return_window_days: int | None,
    return_eligible: bool | None,
) -> dict[str, object]:
    return {
        "decision": decision,
        "summary": summary,
        "final_sale": final_sale,
        "return_window_days": return_window_days,
        "return_eligible": return_eligible,
    }


def _text_from_messages(messages: Sequence[object]) -> str:
    return "\n".join(str(message.content) for message in messages)


def test_sale_hoodie_return_identifies_final_sale_policy() -> None:
    policy_text = (
        "Items marked Sale or Final Sale cannot be returned for fit or change of mind."
    )
    model = FakeChatModel(
        _response(
            decision="not_eligible",
            summary="A sale hoodie cannot be returned for change of mind.",
            final_sale=True,
            return_window_days=30,
            return_eligible=False,
        )
    )
    queried: list[str] = []

    def retrieve(query: str) -> list[Document]:
        queried.append(query)
        return [
            Document(
                page_content=policy_text,
                metadata={"section": "Sale Items", "subsection": "Final-Sale Items"},
            )
        ]

    result = policy_agent(
        {"request_text": "Can I return my sale hoodie?"},
        llm=model,
        retriever=retrieve,
    )["policy_result"]

    assert queried == ["Can I return my sale hoodie?"]
    assert result["final_sale"] is True
    assert result["return_eligible"] is False
    assert "cannot be returned" in result["summary"]
    assert model.schema is not None


def test_normal_return_question_identifies_30_day_rule() -> None:
    model = FakeChatModel(
        _response(
            decision="eligible",
            summary="Standard returns are allowed within 30 calendar days of purchase.",
            final_sale=False,
            return_window_days=30,
            return_eligible=True,
        )
    )
    result = policy_agent(
        {"request_text": "How long do I have to return a regular item?"},
        llm=model,
        retriever=lambda _: [
            Document(
                page_content=(
                    "Standard returns are allowed within 30 calendar days from "
                    "the purchase date."
                ),
                metadata={"section": "Returns and Exchanges", "subsection": "Return Window"},
            )
        ],
    )["policy_result"]

    assert result["return_window_days"] == 30
    assert result["return_eligible"] is True


def test_agent_returns_the_existing_policy_result_shape() -> None:
    model = FakeChatModel(
        _response(
            decision="insufficient_information",
            summary="The policy states a 30-day return window.",
            final_sale=None,
            return_window_days=30,
            return_eligible=None,
        )
    )
    result = policy_agent(
        {"request_text": "What is the return window?"},
        llm=model,
        retriever=lambda _: [Document(page_content="Returns are allowed within 30 days.")],
    )

    policy_result = result["policy_result"]
    assert set(policy_result) == {
        "decision",
        "summary",
        "final_sale",
        "return_window_days",
        "return_eligible",
    }
    assert isinstance(policy_result["decision"], str)
    assert isinstance(policy_result["summary"], str)
    assert isinstance(policy_result["return_window_days"], int)


def test_retrieved_policy_context_is_passed_to_the_llm() -> None:
    policy_text = "Approved refunds are processed within 5–7 business days."
    model = FakeChatModel(
        _response(
            decision="insufficient_information",
            summary="Refund timing is stated after approval.",
            final_sale=None,
            return_window_days=None,
            return_eligible=None,
        )
    )

    policy_agent(
        {"request_text": "When will my refund be processed?"},
        llm=model,
        retriever=lambda _: [
            Document(
                page_content=policy_text,
                metadata={"section": "Refunds", "subsection": "Refund Timing"},
            )
        ],
    )

    assert model.structured.messages is not None
    prompt_text = _text_from_messages(model.structured.messages)
    assert "When will my refund be processed?" in prompt_text
    assert policy_text in prompt_text
    assert "Ground every policy claim in that context" in prompt_text


def test_verified_order_purchase_date_is_passed_to_policy_llm() -> None:
    purchase_date = (date.today() - timedelta(days=40)).isoformat()
    model = FakeChatModel(
        _response(
            decision="not_eligible",
            summary="The purchase was 40 days ago, outside the 30-day return window.",
            final_sale=False,
            return_window_days=30,
            return_eligible=False,
        )
    )

    result = policy_agent(
        {
            "request_text": "Can I return my order #1003?",
            "operations_result": {
                "outcome": "order_details_retrieved",
                "summary": "Order 1003 details retrieved.",
                "order_status": "Delivered",
                "purchase_date": purchase_date,
            },
        },
        llm=model,
        retriever=lambda _: [
            Document(
                page_content="Standard returns are allowed within 30 calendar days from purchase.",
                metadata={"section": "Returns and Exchanges", "subsection": "Return Window"},
            )
        ],
    )

    assert result["policy_result"]["return_eligible"] is False
    prompt_text = _text_from_messages(model.structured.messages)
    assert purchase_date in prompt_text
    assert '"days_since_purchase": 40' in prompt_text
    assert "30 calendar days" in prompt_text


def test_empty_retrieval_returns_safe_result_without_calling_llm() -> None:
    model = FakeChatModel(
        _response(
            decision="eligible",
            summary="This should not be used.",
            final_sale=False,
            return_window_days=30,
            return_eligible=True,
        )
    )
    result: PolicyResult = policy_agent(
        {"request_text": "Can I return this?"},
        llm=model,
        retriever=lambda _: [],
    )["policy_result"]

    assert result["decision"] == "insufficient_information"
    assert result["final_sale"] is None
    assert result["return_window_days"] is None
    assert result["return_eligible"] is None
    assert "could not find relevant policy information" in result["summary"]
    assert model.structured.invocation_count == 0


def test_blank_retrieved_documents_are_treated_as_missing_context() -> None:
    model = FakeChatModel(
        _response(
            decision="eligible",
            summary="This should not be used.",
            final_sale=False,
            return_window_days=30,
            return_eligible=True,
        )
    )

    result = policy_agent(
        {"request_text": "Can I return this?"},
        llm=model,
        retriever=lambda _: [Document(page_content="  \n\t")],
    )["policy_result"]

    assert result["decision"] == "insufficient_information"
    assert model.structured.invocation_count == 0
