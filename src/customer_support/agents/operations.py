"""Operations agent for selecting and running order lookup tools."""

import json
import os
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from customer_support.operations_tools import get_order_details, get_order_status
from customer_support.state import CustomerSupportState, OperationsResult


DEFAULT_CHAT_MODEL = "gpt-4o-mini"
MAX_TOOL_ROUNDS = 3

OPERATIONS_SYSTEM_PROMPT = """You are the Operations Agent for order questions.
Use the available order tools to answer questions about order status, purchase
dates, item details, or reported issues. Never invent order data and never infer
an order ID from dates or other context. If the customer has not provided a
numeric order ID, do not call a tool. Use get_order_status for questions asking
where an order is or its current status. Use get_order_details when the customer
asks for a purchase date, other order details, or asks whether an order qualifies
for a return or exchange. Return the retrieved facts for the Policy Agent to
evaluate; do not decide policy eligibility yourself. Pass the order ID exactly
as provided, without the # marker."""

_DEFAULT_TOOLS: tuple[BaseTool, ...] = (get_order_status, get_order_details)


def operations_agent(
    state: CustomerSupportState,
    *,
    llm: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
) -> dict[str, OperationsResult]:
    """Use tool calling to look up an order and return an OperationsResult.

    The LLM chooses from the existing database tools. Results are mapped
    directly from those structured tool responses rather than parsed from
    arbitrary model prose. ``llm`` and ``tools`` can be injected for tests.
    """

    request_text = (state.get("request_text") or "").strip()
    if not request_text:
        return {"operations_result": _failure_result(
            "missing_order_id", "Please provide a numeric order ID so I can look it up."
        )}

    available_tools = tuple(tools) if tools is not None else _DEFAULT_TOOLS
    tools_by_name = {available_tool.name: available_tool for available_tool in available_tools}
    model = llm or _create_chat_model()
    try:
        tool_calling_model = model.bind_tools(available_tools, tool_choice="auto")
        messages = [
            SystemMessage(content=OPERATIONS_SYSTEM_PROMPT),
            HumanMessage(content=request_text),
        ]
        response = tool_calling_model.invoke(messages)
    except Exception:
        return {"operations_result": _failure_result(
            "tool_error", "The order lookup could not be completed. Please try again."
        )}

    tool_results: list[tuple[str, dict[str, Any]]] = []
    for round_index in range(MAX_TOOL_ROUNDS):
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break

        messages.append(response)
        for call in tool_calls:
            tool_name = call.get("name", "")
            tool_call_id = call.get("id", "")
            arguments = call.get("args") or {}
            if not isinstance(arguments, dict):
                arguments = {}
            selected_tool = tools_by_name.get(tool_name)

            if selected_tool is None:
                result: dict[str, Any] = {
                    "success": False,
                    "order_id": str(arguments.get("order_id", "")),
                    "error": "unsupported_tool",
                }
            elif "order_id" not in arguments:
                result = {
                    "success": False,
                    "order_id": "",
                    "error": "missing_order_id",
                }
            else:
                try:
                    result = selected_tool.invoke(arguments)
                    if not isinstance(result, dict):
                        result = {
                            "success": False,
                            "order_id": str(arguments["order_id"]),
                            "error": "invalid_tool_result",
                        }
                except Exception:
                    result = {
                        "success": False,
                        "order_id": str(arguments.get("order_id", "")),
                        "error": "tool_execution_failed",
                    }

            tool_results.append((tool_name, result))
            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=tool_call_id or f"operations-tool-{round_index}",
                )
            )

        if round_index == MAX_TOOL_ROUNDS - 1:
            break
        try:
            response = tool_calling_model.invoke(messages)
        except Exception:
            # Tool output is still available and can be returned safely.
            break

    if not tool_results:
        return {"operations_result": _failure_result(
            "missing_order_id", "I couldn't identify a numeric order ID in your request."
        )}

    return {"operations_result": _result_from_tool_output(*tool_results[-1])}


def _create_chat_model() -> ChatOpenAI:
    """Create the configured OpenAI chat model."""

    return ChatOpenAI(
        model=os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        temperature=0,
    )


def _result_from_tool_output(
    tool_name: str,
    output: dict[str, Any],
) -> OperationsResult:
    """Map structured tool output into the shared OperationsResult contract."""

    order_id = str(output.get("order_id") or "").strip()
    error = output.get("error")
    if error:
        summaries = {
            "invalid_order_id": "The order ID format is invalid. Please check it and try again.",
            "missing_order_id": "Please provide a numeric order ID so I can look it up.",
            "order_not_found": f"I couldn't find an order with ID {order_id}.",
        }
        outcome = str(error)
        summary = summaries.get(
            outcome, "The order lookup could not be completed. Please try again."
        )
        return _failure_result(outcome, summary)

    if not output.get("success"):
        return _failure_result(
            "tool_error", "The order lookup could not be completed. Please try again."
        )

    order_status = output.get("order_status")
    purchase_date = output.get("purchase_date")
    if tool_name == get_order_status.name:
        return {
            "outcome": "order_status_retrieved",
            "summary": f"Order {order_id} status is {order_status}.",
            "order_status": order_status,
            "purchase_date": None,
        }

    if tool_name == get_order_details.name:
        details = [f"Order {order_id} details retrieved."]
        if order_status:
            details.append(f"Status: {order_status}.")
        if purchase_date:
            details.append(f"Purchase date: {purchase_date}.")
        return {
            "outcome": "order_details_retrieved",
            "summary": " ".join(details),
            "order_status": order_status,
            "purchase_date": purchase_date,
        }

    return _failure_result(
        "tool_error", "The selected tool did not return a supported order result."
    )


def _failure_result(outcome: str, summary: str) -> OperationsResult:
    """Build a predictable non-success OperationsResult."""

    return {
        "outcome": outcome,
        "summary": summary,
        "order_status": None,
        "purchase_date": None,
    }
