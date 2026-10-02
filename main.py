"""A minimal shared state schema for a LangGraph multi-agent workflow."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class MultiAgentState(TypedDict):
    """State shared by the supervisor and worker agents."""

    # LangGraph appends new messages to the existing conversation history.
    messages: Annotated[list[AnyMessage], add_messages]
    # Name of the next agent selected by a supervisor/router node.
    next_agent: str
    # User request or delegated task currently being handled.
    task: str
    # Outputs collected from workers, keyed by agent name.
    agent_outputs: dict[str, str]


if __name__ == "__main__":
    # A small example of the shape; compile a graph separately around this state.
    initial_state: MultiAgentState = {
        "messages": [],
        "next_agent": "supervisor",
        "task": "Describe the task here",
        "agent_outputs": {},
    }
    print(initial_state)
