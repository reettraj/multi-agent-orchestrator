"""Main customer-support graph connecting the Supervisor to specialist agents."""

from collections.abc import Callable, Sequence
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph

from customer_support.agents.escalation import escalation_agent
from customer_support.agents.operations import operations_agent
from customer_support.agents.policy import policy_agent
from customer_support.agents.supervisor import Route, supervisor_agent
from customer_support.state import CustomerSupportState


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
):
    """Build and compile the main graph with optional injectable agent dependencies.

    The Supervisor is the entry point. Its route value selects exactly one
    specialist node, which then exits the graph. This stage deliberately has no
    response-generation node and does not include the isolated HITL workflow.
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

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("policy", policy_node)
    workflow.add_node("operations", operations_node)
    workflow.add_node("escalation", escalation_node)

    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        _selected_route,
        {"policy": "policy", "operations": "operations", "escalation": "escalation"},
    )
    workflow.add_edge("policy", END)
    workflow.add_edge("operations", END)
    workflow.add_edge("escalation", END)
    return workflow.compile()


def _selected_route(state: CustomerSupportGraphState) -> Route:
    """Read, validate, and return the route selected by the Supervisor node."""

    route = state.get("route")
    if route not in {"policy", "operations", "escalation"}:
        raise ValueError("Supervisor did not produce a valid specialist route.")
    return route


# Convenient default graph for application use. Model clients are created lazily
# inside their respective agents only when the graph is invoked.
graph = build_customer_support_graph()
