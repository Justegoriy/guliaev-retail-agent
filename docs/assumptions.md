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
| D4 | Users are defined in `config/users.yaml` and chosen with `--user` (or from a list, or in the web sidebar). No password: authentication is simulated. Demo users: `ceo`, `denim_manager` (Levi's, Diesel, 7 For All Mankind, True Religion), `outdoor_manager` (The North Face, Columbia, Carhartt) |
| - | A brand manager sees the items of their brands, their revenue, and the orders they were part of. A mixed order counts as one order but only with its allowed items and revenue (`num_of_item` is recounted) |

## PII

| # | Assumption |
|---|---|
| D5 | PII fields: `users.first_name`, `last_name`, `email`, `street_address`, `postal_code`, `latitude`, `longitude`, `user_geom` (direct identifiers and precise location, following GDPR/CCPA practice). Not PII: `user_id`, `age`, `gender`, `city`, `state`, `country`, `traffic_source`. The list is in [config/pii_policy.yaml](../config/pii_policy.yaml) |
| D6 | PII may be used inside SQL (joins, filters, `COUNT(DISTINCT email)`) but is never shown, passed to the model in results, logged, traced, cached or saved in reports |
| D7 | No protection against re-identification through small groups (k-anonymity) |
| D8 | Individual customers are shown by id (`Customer #12345`) |

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
| D18 | Models: `gemini-3.8-flash` main, `gemini-3.5-flash-lite` for fallback and auxiliary calls (guard, extraction, summaries, eval judge), `gemini-embedding-001`. All configurable in `.env`. Free-tier limits (not guaranteed by Google) are roughly Flash 5 RPM / 20 RPD and Flash-Lite 15 RPM / 500 RPD, counted per model. With a free key it is practical to set `GEMINI_MODEL` to the Flash-Lite model |
| D19 | Sizing for the HLD: up to 200 executives, up to 20 concurrent sessions, p95 at most 20 s for a simple question, 99.5% availability |
| D20 | The reviewer uses their own GCP project for BigQuery billing (`GOOGLE_CLOUD_PROJECT`) and their own AI Studio key |
| D21 | The conversation is in English: prompts, answers and refusals. The input guard's rules also catch Russian injection and data-changing phrasing, so protection does not depend on the attack's language |
| D22 | No charts in the prototype; they are an extension example in the design |
| D23 | Chat history is stored in SQLite (LangGraph checkpointer) and survives restarts |
| D24 | Business metric definitions (revenue, AOV, margin, return rate, active customer, repeat customer, churn, new customer) are in [config/glossary.yaml](../config/glossary.yaml); the agent states the definition it used |
| D25 | "Now" for relative periods ("last month") is the system date. If there is no data for a period, the agent says so and shows the available range. The public dataset is refreshed daily |
| D26 | One user per CLI process. In the web app each browser tab picks a user; data is isolated per user as in the CLI |
| D27 | Tests with `pytest`, lint with `ruff`, GitHub Actions run both (without network) on every push |
| D28 | A demo web interface (Streamlit, localhost only, no authentication) in addition to the CLI, using the same agent core |

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

## Open questions for OpsFleet

1. **PII list (D5):** please confirm the fields. Should `city` and `age` be treated as PII?
2. **Authentication:** which identity provider will executives use in production (Google Workspace,
   Okta, other)? Where does the mapping from people to brands live (directory groups)?
3. **Company aggregates (D2):** which metrics are acceptable as anonymised company totals? May a
   manager see the share of their categories in total revenue?
4. **Scale and SLA (D19):** please confirm the sizing assumptions.
5. **Model (D18):** which Gemini model and which quota tier will you test with? On a free key the
   main model's daily limit is reached quickly and the agent switches to the fallback.
