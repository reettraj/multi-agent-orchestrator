"""Shared state contract for the customer support workflow."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class CustomerSupportState(TypedDict):
    """Conversation messages shared by workflow nodes."""

    messages: Annotated[list[AnyMessage], add_messages]
