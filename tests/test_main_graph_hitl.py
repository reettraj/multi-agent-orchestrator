"""Integration tests for resumable HITL interrupts in the main graph."""

from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langgraph.types import Command

from customer_support.graph import create_customer_support_graph


class FakeStructuredRunnable:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.invocation_count = 0

    def invoke(self, messages):
        self.invocation_count += 1
        return self.result


class FakeChatModel:
    def __init__(self, result: dict[str, object]) -> None:
        self.structured = FakeStructuredRunnable(result)

    def with_structured_output(self, schema):
        return self.structured


def _state(request_text: str) -> dict[str, object]:
    return {
        "messages": [],
        "request_id": "main-hitl-test",
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


def _config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _escalation_llm() -> FakeChatModel:
    # The agent's deterministic safety triggers will take precedence over this
    # scripted negative assessment for a refund request.
    return FakeChatModel(
        {"required": False, "reason": "No review needed.", "priority": None}
    )


@pytest.mark.parametrize("decision", ["approved", "rejected"])
def test_main_graph_interrupts_and_resumes_with_human_decision(
    tmp_path,
    decision: str,
) -> None:
    checkpoint_path = tmp_path / "main-graph.sqlite"
    thread_id = str(uuid4())
    config = _config(thread_id)
    escalation_llm = _escalation_llm()
    request_text = "My order #1001 arrived completely torn and ruined! I want a refund right now!"

    with create_customer_support_graph(
        checkpoint_path,
        escalation_llm=escalation_llm,
    ) as app:
        paused = app.invoke(_state(request_text), config)
        snapshot = app.get_state(config)

    assert paused["__interrupt__"]
    interrupt_payload = paused["__interrupt__"][0].value
    assert interrupt_payload["type"] == "escalation_approval"
    assert interrupt_payload["request_text"] == request_text
    assert interrupt_payload["escalation"]["required"] is True
    assert interrupt_payload["escalation"]["reason"]
    assert interrupt_payload["escalation"]["priority"] == "high"
    assert "damaged or defective" in interrupt_payload["escalation"]["reason"].lower()
    assert "refund" in interrupt_payload["escalation"]["reason"].lower()
    assert interrupt_payload["allowed_decisions"] == ["approved", "rejected"]
    assert snapshot.values["escalation_result"]["reason"] == interrupt_payload["escalation"]["reason"]
    assert snapshot.values["final_response"] is None
    assert escalation_llm.structured.invocation_count == 1

    # Reopen the SQLite-backed graph and resume the exact paused execution with
    # the same thread ID. The Supervisor and Escalation Agent are not rerun.
    with create_customer_support_graph(
        checkpoint_path,
        escalation_llm=escalation_llm,
    ) as app:
        resumed = app.invoke(
            Command(
                resume={"decision": decision, "reason": "Reviewed by a human agent."}
            ),
            config,
        )

    assert resumed["human_decision"] == {
        "decision": decision,
        "reason": "Reviewed by a human agent.",
    }
    assert resumed["final_response"]
    assert escalation_llm.structured.invocation_count == 1
    if decision == "approved":
        assert "approved" in resumed["final_response"].lower()
        assert "refund has been issued" not in resumed["final_response"].lower()
    else:
        assert "rejected" in resumed["final_response"].lower()
        assert "approved" not in resumed["final_response"].lower()


def test_non_escalation_request_does_not_interrupt_main_graph(tmp_path) -> None:
    checkpoint_path = tmp_path / "main-graph.sqlite"
    thread_id = str(uuid4())
    policy_llm = FakeChatModel(
        {
            "decision": "insufficient_information",
            "summary": "Shipping information is not available in this test policy context.",
            "final_sale": None,
            "return_window_days": None,
            "return_eligible": None,
        }
    )
    with create_customer_support_graph(
        checkpoint_path,
        policy_llm=policy_llm,
        policy_retriever=lambda _: [Document(page_content="Standard orders are dispatched.")],
    ) as app:
        result = app.invoke(
            _state("How long does shipping take?"),
            _config(thread_id),
        )

    assert "__interrupt__" not in result
    assert result["route"] == "policy"
    assert result["human_decision"] is None
    assert result["final_response"] == (
        "Shipping information is not available in this test policy context."
    )


def test_escalation_route_with_no_human_review_continues_to_final_response(tmp_path) -> None:
    checkpoint_path = tmp_path / "main-graph.sqlite"
    supervisor_llm = FakeChatModel(
        {"route": "escalation", "reason": "This is an ambiguous service concern."}
    )
    escalation_llm = FakeChatModel(
        {
            "required": False,
            "reason": "No human escalation trigger was identified.",
            "priority": None,
        }
    )
    with create_customer_support_graph(
        checkpoint_path,
        supervisor_llm=supervisor_llm,
        escalation_llm=escalation_llm,
    ) as app:
        result = app.invoke(
            _state("I have a question about the label on my package."),
            _config(str(uuid4())),
        )

    assert "__interrupt__" not in result
    assert result["route"] == "escalation"
    assert result["escalation_result"]["required"] is False
    assert result["human_decision"] is None
    assert result["final_response"] == "No human escalation trigger was identified."
