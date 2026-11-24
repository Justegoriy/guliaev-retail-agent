# Assumptions and open questions

The assignment leaves several decisions open. The prototype is built on the defaults below; each
can be changed if OpsFleet decides otherwise. Ids (D1, T1...) are referenced from the code
comments and the other documents.

## Data access

| # | Assumption |
|---|---|
| D1 | **Access by brand** (confirmed by OpsFleet). Brand is the only access dimension: a `manager` manages one or more brands and analyses only those (exact `products.brand` values); the mapping is the access table [config/users.yaml](../config/users.yaml). The rewrite applies to every table, so orders, items and customers are those of the user's brands |
| D2 | **Outside their brands, managers see nothing** (confirmed by OpsFleet): no other brands and no company-wide totals, shares or averages; only the `ceo` sees company-wide data. A question that goes beyond the user's brands is answered from what they can see, and the agent says which part is outside their access. Totals of a manager's queries are labelled as their brands' totals |
| D3 | Only the `ceo` has access to all brands and company-wide data |
| D4 | **Access scope from a JWT** (confirmed by OpsFleet): the frontend sends a JWT whose claims carry the user's role and brands; the agent verifies it (signature, issuer, audience, expiry) and enforces access from those claims only. In the prototype the CLI and the web app play the frontend: `--user` (or the web sidebar) issues a token from the access table `config/users.yaml` and verifies it like any other; `--token` / `RETAIL_AGENT_TOKEN` and `?token=` in the web URL accept a token issued elsewhere. HS256 with a local secret (`AUTH_JWT_SECRET` or a generated `data/dev_jwt_secret`). Demo users: `ceo`, `denim_manager` (Levi's, Diesel, 7 For All Mankind, True Religion), `outdoor_manager` (The North Face, Columbia, Carhartt) |
| - | A brand manager sees the items of their brands, their revenue, and the orders they were part of. A mixed order counts as one order but only with its allowed items and revenue (`num_of_item` is recounted) |

## PII

| # | Assumption |
|---|---|
| D5 | **Never shown** (confirmed by OpsFleet): `users.first_name`, `last_name`, `email`, `street_address`, `postal_code`, `latitude`, `longitude`, `user_geom`. **Customer attributes**, shown only in aggregates: `age`, `gender` (also `orders.gender`), `city`, `state`, `country`, `traffic_source` (D7). Customer and order ids may be shown. The lists are in [config/pii_policy.yaml](../config/pii_policy.yaml) and [safety/exposure.py](../src/retail_agent/safety/exposure.py) |
| D6 | PII may be used inside SQL (joins, filters, `COUNT(DISTINCT email)`) but is never shown, and never reaches the LLM: query results are masked before they are passed to the model, as are the user's own messages, logs, traces, caches and saved reports |
| D7 | **Customer attributes only in aggregates** (confirmed by OpsFleet): the SQL Guard rejects any query whose output ties an attribute to one individual: next to a customer or order id, one row per customer or order, or an aggregate over one customer (`AVG(age) GROUP BY user_id`). Filtering by an attribute is allowed (`WHERE state = 'Texas'` with customer ids in the output). Not covered: small groups (a city with one customer); there is no minimum group size (k-anonymity) |
| D8 | Individual customers are shown by id (`Customer #12345`), with their purchases but without attributes |

## Reports and destructive operations

| # | Assumption |
|---|---|
| D9 | **A report is saved only when the user explicitly asks** (confirmed by OpsFleet); the agent never offers to save. A saved report is the question, the answer (Markdown) and the SQL behind it, plus title and entities for search |
| D10 | "Client X" in "delete all reports mentioning Client X" is any entity in the report text: brand, product, category or customer id. Matched against title and content (`LIKE`) and the `entities` the model fills in when saving |
| D11 | **Deletion is a hard delete** (confirmed by OpsFleet): the report row is removed, there is no recovery window and no undo. The audit log (ids, digest, decision, trace id; no report content) is kept for 1 year; older entries are purged when the database is opened |
| D12 | Confirmation: preview, then `Delete N reports? [y/N]`. Only `y`/`yes` confirms; anything else cancels. No stronger confirmation for bulk deletes. The pending action lives until the user's next input; a new question cancels it and is answered right away |
| D13 | No sharing: only the author sees and deletes a report, the CEO included (confirmed by OpsFleet) |
| D14 | Reports are markdown. `export_report` (to `exports/`) exists to demonstrate an extension with confirmation, not as a file-format feature |

## Other defaults

| # | Assumption |
|---|---|
| D15 | The Golden Bucket is "theoretical" in the assignment, so the prototype has a synthetic seed of 15 trios ([data/golden_trios.json](../data/golden_trios.json)) with embeddings retrieval, and updates through `/good` candidates plus analyst review. The full pipeline is in the HLD |
| D16 | User preferences are implemented fully in the prototype (explicit and inferred) |
| D17 | Persona is a text file re-read every turn, global for all users. Preferences take priority for format, the persona sets the tone; guardrails are in the immutable core |
| D18 | **Models** (confirmed by OpsFleet: newer Gemini, free AI Studio key, 2-5 LLM calls per question, fallback between Gemini models): `gemini-3.8-flash` main, `gemini-3.5-flash-lite` for fallback and auxiliary calls (guard, extraction, summaries, eval judge), `gemini-embedding-001`; all configurable in `.env`. At most `MAX_LLM_CALLS_PER_TURN` = 5 model calls per question (input guard, up to 3 agent steps, one post-turn call); the last agent step allowed has no tools and must answer. Retries of a failed call and the switch to the fallback model are not counted as extra calls. Free-tier limits (not guaranteed by Google) are roughly Flash 5 RPM / 20 RPD and Flash-Lite 15 RPM / 500 RPD, counted per model: once Flash's daily quota is used up, the agent continues on Flash-Lite |
| D19 | **Sizing** (confirmed by OpsFleet): tens to a few hundred executives, a few active at once, about 100 active managers a day with about 10 questions each. Targets: 10-30 s for a simple question, 1-2 minutes for a long report; 99.5% availability (our assumption). Estimates in [architecture.md](architecture.md) |
| D20 | The reviewer uses their own GCP project for BigQuery billing (`GOOGLE_CLOUD_PROJECT`) and their own AI Studio key |
| D21 | The conversation is in English: prompts, answers and refusals. The input guard's rules also catch Russian injection and data-changing phrasing, so protection does not depend on the attack's language |
| D22 | No charts in the prototype; they are an extension example in the design |
| D23 | Chat history is stored in SQLite (LangGraph checkpointer) and survives restarts |
| D24 | Business metric definitions (revenue, AOV, margin, return rate, active customer, repeat customer, churn, new customer) are in [config/glossary.yaml](../config/glossary.yaml); the agent states the definition it used |
| D25 | "Now" for relative periods ("last month") is the system date. If there is no data for a period, the agent says so and shows the available range. The public dataset is refreshed daily |
| D26 | One user per CLI process. In the web app each browser tab picks a user; data is isolated per user as in the CLI |
| D27 | Tests with `pytest`, lint with `ruff`, GitHub Actions run both (without network) on every push |
| D28 | A demo web interface (Streamlit, localhost only) in addition to the CLI, using the same agent core; sign-in as in D4 |

Revenue and order definitions used throughout: a "sale" is an order item whose status is not
`Cancelled` or `Returned`; revenue is the sum of `sale_price` over sales; return rate is returned
items over items that were not cancelled.

## Technical decisions

| # | Decision |
|---|---|
| T1 | LangGraph as the agent framework (graph of guard, agent, tools; checkpointer; `interrupt` for confirmations; LangSmith) |
| T2 | Gemini through `langchain-google-genai` in the prototype, Vertex AI in production |
| T3 | SQLite for the prototype stores; JSON + in-memory embeddings for the Golden Bucket |
| T4 | Access control by rewriting SQL with `sqlglot`, not by rejecting queries without a filter |
| T5 | Observability: local JSON logs always, LangSmith optional |
| T6 | All "lite" implementations of design-only requirements are built (Golden Bucket, learning loop, eval, persona) |
| T7 | `src` layout; `requirements.txt` ends with `-e .`, so `pip install -r requirements.txt` from the assignment makes `python -m retail_agent` work |

## Answers from OpsFleet (2026-11-24)

The client answered the earlier open questions; the assumptions above are updated and marked
"confirmed by OpsFleet": access by brand (D1-D3), nothing outside the user's brands (D2), PII and
attributes only in aggregates (D5-D8), hard delete and a one-year audit log (D11), reports only on
request (D9), JWT scopes (D4), models and LLM budget (D18), sizing (D19).

## Open questions for OpsFleet

1. **Identity provider:** which IdP issues the JWT in production (Google Workspace, Okta, other),
   and which claim carries the brands? The agent expects `sub`, `role` and `brands`.
2. **Small groups:** should aggregates over very small groups (a city with one customer) be
   suppressed, and with which minimum size (D7)?
