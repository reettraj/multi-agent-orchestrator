"""Tests for the isolated resumable escalation approval workflow."""

from uuid import uuid4

import pytest
from langgraph.types import Command

from customer_support.hitl_escalation import create_hitl_escalation_workflow


def _state(*, required: bool = True) -> dict[str, object]:
    return {
        "messages": [],
        "request_id": "hitl-test",
        "request_text": "I need a refund for my damaged order.",
        "order_id": "1003",
        "intent": "refund_request",
        "customer_sentiment": "frustrated",
        "policy_result": None,
        "operations_result": None,
        "escalation_result": {
            "required": required,
            "reason": "Damaged-order refund request requires human review.",
            "priority": "high",
        },
        "human_decision": None,
        "final_response": None,
        "errors": [],
    }


def _thread_config() -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": str(uuid4())}}


def test_required_escalation_interrupts_the_graph(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoints.sqlite"
    config = _thread_config()

    with create_hitl_escalation_workflow(checkpoint_path) as workflow:
        result = workflow.invoke(_state(), config)
        snapshot = workflow.get_state(config)

    assert checkpoint_path.is_file()
    assert result["__interrupt__"]
    assert snapshot.tasks
    assert snapshot.tasks[0].interrupts


def test_interrupt_exposes_escalation_reason_and_priority(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoints.sqlite"
    config = _thread_config()

    with create_hitl_escalation_workflow(checkpoint_path) as workflow:
        result = workflow.invoke(_state(), config)
        interrupt_value = result["__interrupt__"][0].value
        saved_state = workflow.get_state(config).values

    assert interrupt_value["type"] == "escalation_approval"
    assert interrupt_value["escalation"]["reason"] == (
        "Damaged-order refund request requires human review."
    )
    assert interrupt_value["escalation"]["priority"] == "high"
    assert "request_text" in interrupt_value
    assert saved_state["escalation_result"]["reason"] == (
        "Damaged-order refund request requires human review."
    )


@pytest.mark.parametrize("decision", ["approved", "rejected"])
def test_graph_resumes_from_checkpoint_with_human_decision(
    tmp_path,
    decision: str,
) -> None:
    checkpoint_path = tmp_path / "checkpoints.sqlite"
    config = _thread_config()

    with create_hitl_escalation_workflow(checkpoint_path) as workflow:
        paused = workflow.invoke(_state(), config)
    assert paused["__interrupt__"]

    # Reopen the workflow against the same SQLite checkpoint store, then resume
    # the pending node rather than submitting its original input again.
    with create_hitl_escalation_workflow(checkpoint_path) as workflow:
        resumed = workflow.invoke(
            Command(resume={"decision": decision, "reason": "Human review complete."}),
            config,
        )

    assert resumed["human_decision"] == {
        "decision": decision,
        "reason": "Human review complete.",
    }


def test_non_escalation_does_not_interrupt(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoints.sqlite"
    config = _thread_config()

    with create_hitl_escalation_workflow(checkpoint_path) as workflow:
        result = workflow.invoke(_state(required=False), config)
        snapshot = workflow.get_state(config)

    assert "__interrupt__" not in result
    assert result["human_decision"] is None
    assert snapshot.tasks == ()
