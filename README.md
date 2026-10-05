# Customer Support Orchestrator

A learning-oriented customer-support backend built with Python and LangGraph. A Supervisor routes each request to a policy, order-operations, or escalation specialist. The graph shares typed state, retrieves store policy text through a local Chroma index, reads sample orders from SQLite, and can pause for a human decision before producing a customer-facing response.

The repository currently provides an interactive command-line interface. It does not provide a web API/UI, deployment configuration, authentication, or automatic refunds.

## What it does

- Answers store-policy questions using retrieved passages from [`policies.md`](policies.md).
- Retrieves order status and purchase date through read-only SQLite tools.
- For order-specific return/exchange eligibility, looks up order facts first, then asks Policy to assess those facts against retrieved rules.
- Flags transactional refund requests, damaged/defective items, strong frustration, and requests for human authorization for escalation review.
- Pauses the main graph for an `approved` or `rejected` human decision when the escalation assessment requires review. Resuming records the decision; approval does not issue a refund.
- Builds the final response deterministically from specialist results and the human decision.

See [docs/architecture.md](docs/architecture.md) for the detailed graph, state, retrieval, persistence, and failure behavior.

## Project layout

```text
.
├── main.py                              # Interactive CLI and persistent graph lifecycle
├── policies.md                          # Fictional store policy source
├── orders.db                            # Sample SQLite order database
├── pyproject.toml                       # Package metadata and dependencies
├── src/customer_support/
│   ├── agents/
│   │   ├── supervisor.py                # Deterministic routing plus structured fallback
│   │   ├── policy.py                    # Policy assessment over retrieved context
│   │   ├── operations.py                # Read-only order tool orchestration
│   │   └── escalation.py                # Escalation assessment and risk overrides
│   ├── graph.py                         # Main graph and SQLite checkpoint factory
│   ├── state.py                         # Shared typed state and result contracts
│   ├── policy_knowledge_base.py         # Policy loading and Markdown-aware chunking
│   ├── policy_retrieval.py              # OpenAI embeddings and Chroma search/index
│   ├── order_database.py                # SQLite schema, explicit seed, and lookup
│   ├── operations_tools.py              # LangChain wrappers for order reads
│   ├── hitl_escalation.py               # Resumable human-approval interrupt
│   └── response_generation.py           # Deterministic response renderer
├── docs/architecture.md                 # In-depth architecture and runtime behavior
└── tests/                               # Unit and graph workflow tests
```

## Requirements and setup

Python 3.11 or newer is required. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The application reads process environment variables directly; it does not load `.env` files. Configure `OPENAI_API_KEY` in the shell that runs the CLI. `OPENAI_CHAT_MODEL` is optional and defaults to `gpt-4o-mini`; policy embeddings currently use `text-embedding-3-small` without an environment override. See [`.env.example`](.env.example) for the variable names. Do not commit real credentials.

PowerShell:

```powershell
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:OPENAI_CHAT_MODEL = "gpt-4o-mini"
```

Bash:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_CHAT_MODEL="gpt-4o-mini"
```

## Run the CLI

```powershell
python main.py
```

At startup, the CLI requires an API key and indexes the policy document. Embeddings are generated through OpenAI, and the local Chroma data is stored under `.chroma/`. Enter a customer message at the prompt. If human review is required, the CLI displays the reason and priority, accepts `approved` or `rejected`, optionally records a note, and resumes the same thread. Type `quit` or `exit` to stop.

The graph checkpoint database is stored at `.checkpoints/customer_support.sqlite`. A unique thread ID is generated for each new CLI request, and the same ID is used when that request resumes. The context-managed graph/checkpointer stays open during the interactive session.

The checked-in `orders.db` supplies the sample records. The CLI does not initialize or seed this database automatically. To create or seed a database explicitly, run `python -m customer_support.order_database` with the package installed; the default target is the repository's `orders.db`. Seeding inserts missing sample rows and preserves existing rows.

## Run the test suite

```powershell
python -m pytest
```

The tests use fake chat models, injected tools, and local/mock embeddings where appropriate. They cover routing, state contracts, retrieval, SQLite tools, order-specific policy follow-up, final response generation, and human interrupt/resume behavior; they do not require live OpenAI calls.

## Boundaries and implementation notes

- The included policies and order records are fictional examples, not a production source of truth.
- Order tools are read-only. No refund, exchange, shipment, or account mutation is performed.
- The human decision is an approval/rejection record for follow-up, not a financial authorization or action.
- `build_customer_support_graph()` defaults to an in-memory checkpointer for in-process use. Use `create_customer_support_graph()` when graph checkpoints must survive graph recreation or process restart.
- The current `CustomerSupportState` declares message and error fields, but the workflow agents do not currently populate a message history or append structured errors; most handled failures are returned as specialist result values.
- The implementation is a foundation for experimentation. It has no authentication, authorization, API boundary, deployment model, or operational monitoring.
