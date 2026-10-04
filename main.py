"""Interactive command-line entry point for the customer-support graph."""

import os
import sys
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from langgraph.types import Command

from customer_support.graph import create_customer_support_graph
from customer_support.policy_retrieval import index_policy_knowledge_base


def _initial_state(request_text: str, request_id: str) -> dict[str, Any]:
    """Create the shared state required to start a new graph execution."""

    return {
        "messages": [],
        "request_id": request_id,
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


def _resume_interrupts(app: Any, result: Mapping[str, Any], config: dict[str, Any]) -> Mapping[str, Any]:
    """Collect a human decision and resume each pending graph interrupt."""

    current = result
    while current.get("__interrupt__"):
        pending = current["__interrupt__"][0]
        payload = pending.value
        escalation = payload.get("escalation", {})
        print("\nHuman review required")
        print(f"Request: {payload.get('request_text', '')}")
        print(f"Reason: {escalation.get('reason', 'No reason provided.')}")
        print(f"Priority: {escalation.get('priority') or 'unspecified'}")

        while True:
            decision = input("Decision (approved/rejected): ").strip().lower()
            if decision in {"approved", "rejected"}:
                break
            print("Enter 'approved' or 'rejected'.")

        reason = input("Review note (optional): ").strip() or None
        current = app.invoke(
            Command(resume={"decision": decision, "reason": reason}),
            config,
        )
    return current


def main() -> int:
    """Initialize policy retrieval and run requests until the user exits."""

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is required. Set it in this shell before starting the CLI.",
            file=sys.stderr,
        )
        return 1

    print("Indexing the local policy knowledge base...")
    try:
        index_policy_knowledge_base()
    except Exception as exc:
        print(f"Could not initialize policy retrieval: {exc}", file=sys.stderr)
        return 1

    print("Customer Support Orchestrator. Enter 'quit' or 'exit' to stop.")
    try:
        with create_customer_support_graph() as app:
            while True:
                try:
                    request_text = input("\nCustomer message: ").strip()
                except EOFError:
                    print()
                    break

                if request_text.lower() in {"quit", "exit"}:
                    break
                if not request_text:
                    continue

                thread_id = str(uuid4())
                config = {"configurable": {"thread_id": thread_id}}
                try:
                    result = app.invoke(
                        _initial_state(request_text, thread_id),
                        config,
                    )
                    result = _resume_interrupts(app, result, config)
                    response = result.get("final_response")
                    print(f"\nAssistant: {response or 'No response was generated.'}")
                except Exception as exc:
                    print(f"Request failed: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"Could not start the customer-support graph: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
