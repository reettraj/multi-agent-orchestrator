# Customer Support Orchestrator

A modular customer-support backend built with LangGraph. A Supervisor routes each request to specialist agents, which share typed state and produce a grounded customer-facing response. The project demonstrates policy retrieval, deterministic order tools, escalation decisions, and resumable human review. It currently provides a command-line entry point; a web UI and deployment setup are out of scope.

## Architecture

- **Shared state** carries the request, messages, order identifier, structured policy/operations/escalation results, human decision, final response, and accumulated errors.
- **Supervisor Agent** chooses `policy`, `operations`, or `escalation`. Order-specific return-eligibility requests are routed through Operations before Policy evaluates the order facts.
- **Policy Agent and RAG** load `policies.md`, split it by Markdown headings and chunk size, embed the chunks with OpenAI `text-embedding-3-small`, and search a local Chroma collection. The Policy Agent grounds its structured result in retrieved policy context.
- **Operations Agent** uses LangChain tools backed by the SQLite order database. The database layer owns persistence and queries; the agent selects tools and formats the result.
- **Escalation Agent** identifies requests that need human review, including transactional refunds, damaged items, and strong customer frustration.
- **Human-in-the-loop (HITL)** pauses the main graph when escalation is required. A reviewer can approve or reject, and the same graph execution resumes from its SQLite checkpoint.
- **Final response generation** turns the specialist result and any human decision into a concise response. Approval does not mean a refund has been issued.

The graph normally follows `Supervisor → specialist → Final Response`. For an order return check it follows `Supervisor → Operations → Policy → Final Response`; for an escalated request it may pause at HITL before Final Response.

## Supported end-to-end scenarios

1. A sale-item fit/return question routes to Policy and explains the final-sale rule.
2. An order-status request for order `1002` routes to Operations and retrieves its `Processing` status.
3. An order `1003` return-eligibility question retrieves the purchase date, then Policy evaluates it against the 30-day return window.
4. A damaged-order refund request routes to Escalation, pauses for human approval or rejection, then resumes and responds without claiming a refund was issued.

## Project structure

```text
.
├── main.py                         # Interactive command-line entry point
├── policies.md                     # Fictional store policy source document
├── orders.db                       # Seeded SQLite sample orders
├── src/customer_support/
│   ├── agents/
│   │   ├── supervisor.py           # Request routing
│   │   ├── policy.py               # Grounded policy assessment
│   │   ├── operations.py           # Order tool selection
│   │   └── escalation.py           # Human-review decision
│   ├── graph.py                    # Main LangGraph and persistent HITL support
│   ├── state.py                    # Shared state and result types
│   ├── policy_knowledge_base.py    # Policy loading and chunking
│   ├── policy_retrieval.py         # Chroma indexing and semantic search
│   ├── order_database.py           # SQLite schema, seed, and access
│   ├── operations_tools.py         # LangChain order lookup tools
│   ├── hitl_escalation.py          # Reusable interrupt/resume node
│   └── response_generation.py      # Grounded final response renderer
└── tests/                          # Unit and graph integration tests
```

## Requirements and setup

Python 3.11 or newer is required. From the repository root, create and activate a virtual environment, then install the project and test dependency:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Set the environment variables in the shell that will run the application. The names and defaults are shown in [`.env.example`](.env.example). The application reads process environment variables directly; it does not automatically load a `.env` file.

PowerShell example:

```powershell
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:OPENAI_CHAT_MODEL = "gpt-4o-mini"
```

Bash example:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_CHAT_MODEL="gpt-4o-mini"
```

`OPENAI_API_KEY` is required for live chat completions and policy embeddings. `OPENAI_CHAT_MODEL` is optional and defaults to `gpt-4o-mini`. The policy embedding model is currently `text-embedding-3-small`. Never commit a real key.

## Run the command-line assistant

```powershell
python main.py
```

The CLI indexes the policy document in the local Chroma collection before starting the graph. Enter a customer request at the prompt. If the graph pauses for HITL, review the displayed reason and priority, then enter `approved` or `rejected`; the CLI resumes the same request using its thread ID. Enter `quit` or `exit` to leave.

The first policy-indexing run makes embedding API calls and creates local Chroma data under `.chroma/`. The graph checkpoint is stored under `.checkpoints/`; both directories are ignored by Git. The included `orders.db` contains deterministic fictional test orders.

## Run tests

```powershell
python -m pytest
```

The current suite contains **84 tests**. It uses fake models and local/mock embeddings, so tests do not require an OpenAI API key or make live model calls.

## Scope

This repository is a learning-oriented backend foundation. It does not include a web interface, deployment configuration, production customer data, or automatic refund issuance.
