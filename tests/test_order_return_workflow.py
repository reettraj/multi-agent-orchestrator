"""Integration coverage for Operations-to-Policy return evaluation."""

from datetime import date
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from customer_support.graph import build_customer_support_graph


class FakePolicyRunnable:
    def __init__(self) -> None:
        self.messages = None
        self.invocation_count = 0

    def invoke(self, messages):
        self.messages = messages
        self.invocation_count += 1
        return {
            "decision": "not_eligible",
            "summary": "The order is outside the 30-day return window.",
            "final_sale": False,
            "return_window_days": 30,
            "return_eligible": False,
        }


class FakePolicyLLM:
    def __init__(self) -> None:
        self.structured = FakePolicyRunnable()
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured


class FakeOperationsLLM:
    def __init__(self) -> None:
        self.responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_order_details",
                        "args": {"order_id": "1003"},
                        "id": "order-details-1003",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Order details retrieved."),
        ]
        self.invocations = []
        self.bound_tools = []

    def bind_tools(self, tools, *, tool_choice):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        self.invocations.append(list(messages))
        return self.responses.pop(0)


def test_order_1003_runs_operations_then_policy_with_verified_order_context() -> None:
    operations_llm = FakeOperationsLLM()
    policy_llm = FakePolicyLLM()
    retrieval_queries: list[str] = []

    def retrieve_policy(query: str) -> list[Document]:
        retrieval_queries.append(query)
        return [
            Document(
                page_content=(
                    "Standard returns are allowed within 30 calendar days from the purchase date."
                ),
                metadata={
                    "section": "Returns and Exchanges",
                    "subsection": "Return Window",
                },
            )
        ]

    app = build_customer_support_graph(
        operations_llm=operations_llm,
        policy_llm=policy_llm,
        policy_retriever=retrieve_policy,
    )
    result = app.invoke(
        {
            "messages": [],
            "request_id": "return-workflow-1003",
            "request_text": "Can I return my order #1003?",
            "order_id": None,
            "intent": None,
            "customer_sentiment": None,
            "policy_result": None,
            "operations_result": None,
            "escalation_result": None,
            "human_decision": None,
            "final_response": None,
            "errors": [],
            "route": None,
            "route_reason": None,
        },
        {"configurable": {"thread_id": str(uuid4())}},
    )

    operation_result = result["operations_result"]
    days_since_purchase = (
        date.today() - date.fromisoformat(operation_result["purchase_date"])
    ).days
    assert result["route"] == "operations"
    assert operation_result["outcome"] == "order_details_retrieved"
    assert operation_result["purchase_date"] == "2026-08-25"
    assert days_since_purchase > 30
    assert retrieval_queries == ["Can I return my order #1003?"]
    operations_prompt = " ".join(operations_llm.invocations[0][0].content.split())
    assert "order qualifies for a return or exchange" in operations_prompt
    assert len(operations_llm.invocations) == 2

    assert policy_llm.structured.invocation_count == 1
    assert policy_llm.schema is not None
    policy_prompt = "\n".join(
        str(message.content) for message in policy_llm.structured.messages
    )
    assert "2026-08-25" in policy_prompt
    assert f'"days_since_purchase": {days_since_purchase}' in policy_prompt
    assert "30 calendar days" in policy_prompt
    assert result["policy_result"]["return_window_days"] == 30
    assert result["policy_result"]["return_eligible"] is False
    assert result["final_response"] == "The order is outside the 30-day return window."
    assert result["escalation_result"] is None
