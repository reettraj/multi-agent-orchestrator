# Customer Support Orchestrator

A modular Python foundation for a four-agent customer support system built with LangGraph.

## Project layout

- `src/customer_support/` contains the application package.
- `src/customer_support/agents/` reserves one module for each planned agent: Supervisor, Policy, Operations, and Escalation.
- `tests/` contains the initial checks for the shared state and graph module.

## Setup

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Run the foundation checks with:

```powershell
python -m pytest
```

This stage establishes package structure, configuration, and the shared state contract. Agent behavior, graph routing, retrieval, database access, tools, and human review are intentionally outside this stage.
