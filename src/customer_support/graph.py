"""Main customer-support graph and resumable escalation review flow."""

from contextlib import contextmanager
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypedDict
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from customer_support.agents.escalation import escalation_agent
from customer_support.agents.operations import operations_agent
from customer_support.agents.policy import policy_agent
from customer_support.agents.supervisor import (
    Route,
    requires_order_policy_lookup,
    supervisor_agent,
)
from customer_support.hitl_escalation import human_approval_node
from customer_support.response_generation import generate_final_response
from customer_support.state import CustomerSupportState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / ".checkpoints" / "customer_support.sqlite"


class CustomerSupportGraphState(CustomerSupportState):
    """Shared workflow state plus the Supervisor's internal routing decision."""

    route: Route | None
    route_reason: str | None


def build_customer_support_graph(
    *,
    supervisor_llm: BaseChatModel | None = None,
    policy_llm: BaseChatModel | None = None,
    policy_retriever: Callable[[str], Sequence[Document]] | None = None,
    operations_llm: BaseChatModel | None = None,
    operations_tools: Sequence[BaseTool] | None = None,
    escalation_llm: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Build and compile the main graph with optional injectable agent dependencies.

    The Supervisor is the entry point. Its route value selects exactly one
    specialist node. Escalation results requiring human review pause at the
    reusable HITL approval node. After any required decision, the graph creates
    its final response and ends.
    """

    workflow = StateGraph(CustomerSupportGraphState)

    def supervisor_node(state: CustomerSupportGraphState) -> dict[str, object]:
        decision = supervisor_agent(state, llm=supervisor_llm)
        return {"route": decision["route"], "route_reason": decision["reason"]}

    def policy_node(state: CustomerSupportGraphState) -> dict[str, object]:
        return policy_agent(state, llm=policy_llm, retriever=policy_retriever)

    def operations_node(state: CustomerSupportGraphState) -> dict[str, object]:
        return operations_agent(state, llm=operations_llm, tools=operations_tools)

    def escalation_node(state: CustomerSupportGraphState) -> dict[str, object]:
        return escalation_agent(state, llm=escalation_llm)

    def final_response_node(state: CustomerSupportGraphState) -> dict[str, str]:
        return generate_final_response(state)

    def human_approval_node_wrapper(
        state: CustomerSupportGraphState,
    ) -> dict[str, object]:
        return human_approval_node(state)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("policy", policy_node)
    workflow.add_node("operations", operations_node)
    workflow.add_node("escalation", escalation_node)
    workflow.add_node("human_approval", human_approval_node_wrapper)
    workflow.add_node("final_response", final_response_node)

    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        _selected_route,
        {"policy": "policy", "operations": "operations", "escalation": "escalation"},
    )
    workflow.add_edge("policy", "final_response")
    workflow.add_conditional_edges(
        "operations",
        _policy_follow_up_required,
        {"policy": "policy", "final_response": "final_response"},
    )
    # Policy normally ends after its first assessment. For an order-specific
    # return check, Operations has already run and this second Policy pass ends
    # directly at final response.
    workflow.add_conditional_edges(
        "escalation",
        _approval_required,
        {"human_approval": "human_approval", "final_response": "final_response"},
    )
    workflow.add_edge("human_approval", "final_response")
    workflow.add_edge("final_response", END)
    return workflow.compile(
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver()
    )


@contextmanager
def create_customer_support_graph(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    **agent_dependencies: object,
) -> Iterator[CompiledStateGraph]:
    """Yield a main graph backed by a persistent local SQLite checkpoint store.

    Keep the context open while invoking and resuming runs. Pass the same
    ``configurable.thread_id`` to each invocation for the same request. The
    store makes pending interrupts resumable after the graph is recreated.
    """

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as checkpointer:
        yield build_customer_support_graph(
            checkpointer=checkpointer,
            **agent_dependencies,  # type: ignore[arg-type]
        )


def _selected_route(state: CustomerSupportGraphState) -> Route:
    """Read, validate, and return the route selected by the Supervisor node."""

    route = state.get("route")
    if route not in {"policy", "operations", "escalation"}:
        raise ValueError("Supervisor did not produce a valid specialist route.")
    return route


def _approval_required(state: CustomerSupportGraphState) -> str:
    escalation = state.get("escalation_result")
    if escalation and escalation.get("required") is True:
        return "human_approval"
    return "final_response"


def _policy_follow_up_required(state: CustomerSupportGraphState) -> str:
    operations_result = state.get("operations_result")
    has_order_details = bool(
        operations_result
        and operations_result.get("outcome") == "order_details_retrieved"
        and operations_result.get("purchase_date")
    )
    if has_order_details and requires_order_policy_lookup(
        state, state.get("request_text")
    ):
        return "policy"
    return "final_response"


# Convenient default graph for in-process use. Use
# create_customer_support_graph() when checkpoint persistence across graph
# recreation or process restarts is needed. Model clients are created lazily.
graph = build_customer_support_graph()
