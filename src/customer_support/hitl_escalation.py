"""Standalone, resumable LangGraph workflow for human escalation decisions."""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from customer_support.state import CustomerSupportState, EscalationResult, HumanDecision


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / ".checkpoints" / "hitl_escalation.sqlite"


class HumanApprovalResponse(BaseModel):
    """Decision submitted when resuming the escalation interrupt."""

    decision: Literal["approved", "rejected"] = Field(
        description="Whether the human approves or rejects the escalation"
    )
    reason: str | None = Field(
        default=None,
        description="Optional reason for the human decision",
    )


@contextmanager
def create_hitl_escalation_workflow(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
) -> Iterator[CompiledStateGraph]:
    """Open the local SQLite checkpointer and yield a compiled HITL graph.

    Keep the context open while invoking and resuming a workflow. The checkpoint
    database preserves interrupt state across invocations and graph recreation.
    """

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as checkpointer:
        builder = StateGraph(CustomerSupportState)
        builder.add_node("human_approval", human_approval_node)
        builder.add_edge(START, "human_approval")
        builder.add_edge("human_approval", END)
        yield builder.compile(checkpointer=checkpointer)


def human_approval_node(state: CustomerSupportState) -> dict[str, HumanDecision | None]:
    """Reusable approval node that pauses only when escalation is required."""

    escalation: EscalationResult | None = state.get("escalation_result")
    if not escalation or escalation.get("required") is not True:
        return {"human_decision": state.get("human_decision")}

    human_response = interrupt(
        {
            "type": "escalation_approval",
            "request_text": state.get("request_text", ""),
            "escalation": {
                "required": escalation["required"],
                "reason": escalation["reason"],
                "priority": escalation["priority"],
            },
            "allowed_decisions": ["approved", "rejected"],
            "instructions": "Review the escalation reason and submit approved or rejected.",
        },
        response_schema=HumanApprovalResponse,
    )
    decision = HumanApprovalResponse.model_validate(human_response)
    result: HumanDecision = {
        "decision": decision.decision,
        "reason": decision.reason,
    }
    return {"human_decision": result}
