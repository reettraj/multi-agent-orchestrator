"""Integration tests for Supervisor-to-agent graph routing."""

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from customer_support.graph import build_customer_support_graph


class FakeStructuredModel:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.invocation_count = 0

    def invoke(self, messages):
        self.invocation_count += 1
        return self.response


class FakeStructuredChatModel:
    def __init__(self, response: dict[str, object]) -> None:
        self.structured = FakeStructuredModel(response)
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured


class FakeToolCallingLLM:
    def __init__(self, *responses: AIMessage) -> None:
        self.responses = list(responses)
        self.invocation_count = 0
        self.bound_tools = []

    def bind_tools(self, tools, *, tool_choice):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        self.invocation_count += 1
        return self.responses.pop(0)


def _state(request_text: str) -> dict[str, object]:
    return {
        "messages": [],
        "request_id": "graph-test",
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
        "route": None,
        "route_reason": None,
    }


def _order_status_llm() -> FakeToolCallingLLM:
    return FakeToolCallingLLM(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_order_status",
                    "args": {"order_id": "1002"},
                    "id": "graph-order-status",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="Order lookup complete."),
    )


def test_policy_request_runs_policy_agent_only() -> None:
    policy_llm = FakeStructuredChatModel(
        {
            "decision": "not_eligible",
            "summary": "Sale items are final sale.",
            "final_sale": True,
            "return_window_days": 30,
            "return_eligible": False,
        }
    )
    operations_llm = _order_status_llm()
    escalation_llm = FakeStructuredChatModel(
        {"required": False, "reason": "No escalation required.", "priority": None}
    )
    app = build_customer_support_graph(
        policy_llm=policy_llm,
        policy_retriever=lambda _: [
            Document(page_content="Sale items are final sale.", metadata={"section": "Sale Items"})
        ],
        operations_llm=operations_llm,
        escalation_llm=escalation_llm,
    )

    result = app.invoke(_state("Can I return my discounted hoodie?"))

    assert result["route"] == "policy"
    assert result["policy_result"]["final_sale"] is True
    assert result["final_response"] == "Sale items are final sale."
    assert result["operations_result"] is None
    assert result["escalation_result"] is None
    assert policy_llm.structured.invocation_count == 1
    assert operations_llm.invocation_count == 0
    assert escalation_llm.structured.invocation_count == 0


def test_order_status_request_runs_operations_agent_only() -> None:
    operations_llm = _order_status_llm()
    policy_llm = FakeStructuredChatModel(
        {
            "decision": "insufficient_information",
            "summary": "Unused.",
            "final_sale": None,
            "return_window_days": None,
            "return_eligible": None,
        }
    )
    escalation_llm = FakeStructuredChatModel(
        {"required": False, "reason": "No escalation required.", "priority": None}
    )
    app = build_customer_support_graph(
        policy_llm=policy_llm,
        policy_retriever=lambda _: [],
        operations_llm=operations_llm,
        escalation_llm=escalation_llm,
    )

    result = app.invoke(
        _state("Can you tell me where my order #1002 is? I ordered it yesterday.")
    )

    assert result["route"] == "operations"
    assert result["operations_result"]["order_status"] == "Processing"
    assert result["final_response"] == "The current status of your order is Processing."
    assert "purchase date" not in result["final_response"].lower()
    assert result["policy_result"] is None
    assert result["escalation_result"] is None
    assert len(operations_llm.bound_tools) == 2
    assert operations_llm.invocation_count == 2
    assert policy_llm.structured.invocation_count == 0
    assert escalation_llm.structured.invocation_count == 0


def test_damaged_refund_request_runs_escalation_agent_only() -> None:
    escalation_llm = FakeStructuredChatModel(
        {
            "required": False,
            "reason": "The request needs human review.",
            "priority": None,
        }
    )
    policy_llm = FakeStructuredChatModel(
        {
            "decision": "insufficient_information",
            "summary": "Unused.",
            "final_sale": None,
            "return_window_days": None,
            "return_eligible": None,
        }
    )
    operations_llm = _order_status_llm()
    app = build_customer_support_graph(
        policy_llm=policy_llm,
        policy_retriever=lambda _: [],
        operations_llm=operations_llm,
        escalation_llm=escalation_llm,
    )

    result = app.invoke(_state("My order arrived damaged and I want a refund."))

    assert result["route"] == "escalation"
    assert result["escalation_result"]["required"] is True
    assert result["escalation_result"]["priority"] == "high"
    assert result["final_response"] == (
        "Human review is required: The customer is requesting a refund, which needs human review. "
        "The customer reports a damaged or defective item that needs human assessment."
    )
    assert result["policy_result"] is None
    assert result["operations_result"] is None
    assert escalation_llm.structured.invocation_count == 1
    assert policy_llm.structured.invocation_count == 0
    assert operations_llm.invocation_count == 0


def test_graph_uses_supervisor_route_as_its_only_dispatch_value() -> None:
    policy_llm = FakeStructuredChatModel(
        {
            "decision": "insufficient_information",
            "summary": "An informational sizing policy question.",
            "final_sale": None,
            "return_window_days": None,
            "return_eligible": None,
        }
    )
    app = build_customer_support_graph(
        policy_llm=policy_llm,
        policy_retriever=lambda _: [Document(page_content="Use the size chart.")],
    )

    result = app.invoke(_state("How should I use the size chart?"))

    assert result["route"] == "policy"
    assert result["route_reason"]
    assert result["policy_result"]["decision"] == "insufficient_information"
    assert result["final_response"] == "An informational sizing policy question."


def test_final_response_contains_only_order_facts_in_operations_result() -> None:
    operations_llm = _order_status_llm()
    app = build_customer_support_graph(operations_llm=operations_llm)

    result = app.invoke(_state("Where is order #1002?"))

    assert result["operations_result"]["purchase_date"] is None
    assert result["final_response"] == "The current status of your order is Processing."
    assert "delivered" not in result["final_response"].lower()
    assert "purchase date" not in result["final_response"].lower()
