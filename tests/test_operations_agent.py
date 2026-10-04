"""Tests for tool selection and structured Operations Agent results."""

import json

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from customer_support.agents.operations import operations_agent
from customer_support.state import OperationsResult


class FakeToolCallingLLM:
    """Return scripted tool calls without making provider API requests."""

    def __init__(self, *responses: AIMessage) -> None:
        self.responses = list(responses)
        self.bound_tools = []
        self.tool_choice = None
        self.invocations = []

    def bind_tools(self, tools, *, tool_choice):
        self.bound_tools = list(tools)
        self.tool_choice = tool_choice
        return self

    def invoke(self, messages):
        self.invocations.append(list(messages))
        return self.responses.pop(0)


def _tool_call(name: str, order_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": {"order_id": order_id},
                "id": f"call-{name}",
                "type": "tool_call",
            }
        ],
    )


def _finished_llm(name: str, order_id: str) -> FakeToolCallingLLM:
    return FakeToolCallingLLM(_tool_call(name, order_id), AIMessage(content="Lookup complete."))


def test_order_1002_uses_status_tool_and_returns_processing() -> None:
    llm = _finished_llm("get_order_status", "1002")

    result = operations_agent(
        {
            "request_text": (
                "Can you tell me where my order #1002 is? I ordered it yesterday."
            )
        },
        llm=llm,
    )["operations_result"]

    assert llm.tool_choice == "auto"
    assert {tool.name for tool in llm.bound_tools} == {
        "get_order_status",
        "get_order_details",
    }
    assert len(llm.invocations) == 2
    tool_messages = [
        message
        for message in llm.invocations[1]
        if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 1
    assert json.loads(tool_messages[0].content)["order_status"] == "Processing"
    assert result["outcome"] == "order_status_retrieved"
    assert result["order_status"] == "Processing"
    assert result["purchase_date"] is None


def test_order_1003_details_include_purchase_date() -> None:
    llm = _finished_llm("get_order_details", "1003")

    result = operations_agent(
        {"request_text": "What date did I purchase order 1003?"}, llm=llm
    )["operations_result"]

    assert result["outcome"] == "order_details_retrieved"
    assert result["purchase_date"] is not None
    assert result["order_status"] == "Delivered"
    tool_message = next(
        message
        for message in llm.invocations[1]
        if isinstance(message, ToolMessage)
    )
    assert json.loads(tool_message.content)["purchase_date"] == result["purchase_date"]


def test_return_eligibility_lookup_uses_order_details_tool() -> None:
    llm = _finished_llm("get_order_details", "1003")

    result = operations_agent(
        {"request_text": "Can I return my order #1003?"},
        llm=llm,
    )["operations_result"]

    assert result["outcome"] == "order_details_retrieved"
    assert result["purchase_date"] is not None
    prompt = " ".join(llm.invocations[0][0].content.split())
    assert "order qualifies for a return or exchange" in prompt


def test_unknown_order_is_reported_without_inventing_details() -> None:
    llm = _finished_llm("get_order_status", "999999")

    result = operations_agent(
        {"request_text": "Where is order #999999?"}, llm=llm
    )["operations_result"]

    assert result["outcome"] == "order_not_found"
    assert "999999" in result["summary"]
    assert result["order_status"] is None
    assert result["purchase_date"] is None


def test_malformed_order_id_is_handled_safely() -> None:
    llm = _finished_llm("get_order_details", "ABC-1003")

    result = operations_agent(
        {"request_text": "What happened with order #ABC-1003?"}, llm=llm
    )["operations_result"]

    assert result["outcome"] == "invalid_order_id"
    assert "invalid" in result["summary"].lower()
    assert result["purchase_date"] is None


def test_missing_order_id_returns_safe_result_without_tool_calls() -> None:
    llm = FakeToolCallingLLM(AIMessage(content="Please provide an order number."))

    result = operations_agent(
        {"request_text": "Where is my order?"}, llm=llm
    )["operations_result"]

    assert result["outcome"] == "missing_order_id"
    assert result["order_status"] is None
    assert result["purchase_date"] is None
    assert len(llm.invocations) == 1


def test_tool_execution_failure_returns_safe_result() -> None:
    def fail_lookup(order_id: str) -> dict[str, str]:
        raise RuntimeError("simulated database failure")

    broken_tool = StructuredTool.from_function(
        func=fail_lookup,
        name="get_order_status",
        description="Simulate an unavailable order lookup.",
    )
    llm = _finished_llm("get_order_status", "1002")

    result = operations_agent(
        {"request_text": "Where is order #1002?"},
        llm=llm,
        tools=[broken_tool],
    )["operations_result"]

    assert result["outcome"] == "tool_execution_failed"
    assert result["order_status"] is None
    assert result["purchase_date"] is None


def test_final_result_matches_operations_result_structure() -> None:
    llm = _finished_llm("get_order_status", "1002")

    result: OperationsResult = operations_agent(
        {"request_text": "Where is order #1002?"}, llm=llm
    )["operations_result"]

    assert set(result) == {"outcome", "summary", "order_status", "purchase_date"}
    assert isinstance(result["outcome"], str)
    assert isinstance(result["summary"], str)
    assert result["order_status"] == "Processing"
