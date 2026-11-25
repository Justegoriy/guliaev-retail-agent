# Architecture

This document is the high-level design (HLD) of the Retail Analytics Agent: the production
architecture, the main flows as sequence diagrams, where data lives and how the components talk
to each other. How each requirement is solved, and what the prototype implements, is in
[technical_design.md](technical_design.md). Assumptions behind the numbers are in
[assumptions.md](assumptions.md).

Contents:

1. [Production architecture](#1-production-architecture)
2. [Services: what, why, alternatives](#2-services-what-why-alternatives)
3. [Answering a question](#3-answering-a-question)
4. [Deleting reports with confirmation](#4-deleting-reports-with-confirmation)
5. [Learning loop](#5-learning-loop)
6. [Observability and the path of PII](#6-observability-and-the-path-of-pii)
7. [Prototype vs production](#7-prototype-vs-production)
8. [Data storage](#8-data-storage)
9. [Component interaction](#9-component-interaction)

Sizing (D19, confirmed by OpsFleet): tens to a few hundred executives, about 100 active managers
a day asking about 10 questions each, only a few active at once. Targets: 10-30 s for a simple
question, 1-2 minutes for a long report. What that means for the design:

| Quantity | Estimate | How it was derived |
|---|---|---|
| Questions | about 1,000 a day; about 2 a minute on average in an 8-hour day, about 6 a minute at a 3x peak | 100 managers x 10 questions |
| Busy sessions at the peak | about 3 for simple questions (30 s), up to about 12 if all were long reports (2 min) | 6 questions a minute x time per question |
| LLM calls | at most 5 per question (input guard, up to 3 agent steps, one post-turn call), so at most 5,000 a day and about 30 a minute at the peak | the per-turn budget, see technical design R5 |
| Tokens per question | median 11k in / 0.35k out, p90 21k / 1.2k | prototype eval run of 2026-11-23 |
| LLM cost | about $10 a day at the median, $21 a day at the p90 (Flash prices, $0.75 / $3.75 per 1M tokens; auxiliary calls on Flash-Lite cost less) | 1,000 questions x tokens |
| BigQuery | about 10 MB per query on `thelook_ecommerce`, under 0.1 TB a day; capped at 2 GB per query and 10 GB per question | prototype logs |

So the load is small: one Cloud Run service with a minimum of one instance covers the peak, and
the binding constraint is the LLM quota, not compute. On a free AI Studio key (Flash about 5
requests a minute and 20 a day, Flash-Lite about 15 and 500) the prototype answers a handful of
questions a day on Flash, about 6 since each uses up to 3 Flash calls, and then switches to
Flash-Lite automatically (D18); production uses Vertex AI with a paid quota. Availability target:
99.5%.

---

## 1. Production architecture

```mermaid
flowchart TB
    subgraph Clients
        CLI[CLI]
        WEB[Web app]
        SLACK[Slack / Teams bot]
        ADMIN[Admin UI<br/>persona, Golden Bucket audit]
    end

    subgraph Edge
        LB[HTTPS Load Balancer + Cloud Armor]
        IAP[Identity-Aware Proxy<br/>SSO: Google Workspace / Okta]
    end

    subgraph Compute["Compute (Cloud Run, private VPC)"]
        AGENT[Agent service<br/>LangGraph: input guard, agent, tools,<br/>SQL Guard, PII masker, confirmation]
        ADMINAPI[Admin API]
        WORKERS[Workers<br/>preference signals, feedback,<br/>Golden candidates]
        INDEXER[Golden indexer<br/>Cloud Run Job]
        EVAL[Eval runner<br/>Cloud Run Job]
    end

    subgraph AI["Vertex AI"]
        GEMINI[Gemini Flash<br/>main]
        GEMINIL[Gemini Flash-Lite<br/>fallback, guard, extraction, judge]
        EMB[Embeddings]
        VS[Vector Search<br/>Golden trios index]
    end

    subgraph Data
        BQ[(BigQuery thelook_ecommerce<br/>read-only, row access policies,<br/>policy tags, authorized views)]
        GCS[(GCS<br/>Golden Bucket raw trios,<br/>report artifacts, exports)]
        FS[(Firestore<br/>reports, conversations, checkpoints,<br/>preferences, pending actions,<br/>persona versions, audit)]
        REDIS[(Memorystore Redis<br/>rate limits, profile cache,<br/>query cache, circuit state)]
    end

    subgraph Platform
        SM[Secret Manager]
        DLP[Cloud DLP]
        PS[[Pub/Sub]]
        OBS[Cloud Logging / Trace / Monitoring<br/>+ LangSmith, BigQuery log sink]
        CICD[Cloud Build / GitHub Actions<br/>tests, eval gate, canary]
    end

    CLI & WEB & SLACK --> LB --> IAP --> AGENT
    ADMIN --> LB
    IAP --> ADMINAPI

    AGENT --> GEMINI
    AGENT --> GEMINIL
    AGENT --> EMB --> VS
    AGENT -->|SQL as the user's group| BQ
    AGENT --> FS
    AGENT --> REDIS
    AGENT --> DLP
    AGENT --> SM
    AGENT -->|events| PS

    PS --> WORKERS
    WORKERS --> FS
    WORKERS --> GEMINIL
    GCS -->|object finalize| PS
    PS --> INDEXER
    INDEXER --> EMB
    INDEXER --> VS

    ADMINAPI --> FS
    ADMINAPI --> GCS
    ADMINAPI -->|publish persona / approve trio| PS

    CICD --> EVAL
    EVAL --> AGENT

    AGENT -.-> OBS
    WORKERS -.-> OBS
    INDEXER -.-> OBS
```

How to read it:

- **One stateless agent service.** All conversation state (messages, interrupts, analysis
  context) lives in a checkpointer backed by Firestore, so any Cloud Run instance can serve any
  turn and a pending confirmation survives a redeploy.
- **The synchronous path is short:** client, IAP, agent service, then Vertex AI, BigQuery and
  Firestore. Everything that is not needed to answer the user (extracting preference signals,
  feedback, indexing new trios, eval) is asynchronous through Pub/Sub.
- **Safety is enforced twice:** in the agent (SQL Guard, ACL rewrite, PII masking, input guard)
  and in the data platform (BigQuery row access policies, policy tags on PII columns, authorized
  views). A bug in the agent cannot leak data the user's BigQuery identity cannot read.

---

## 2. Services: what, why, alternatives

| Building block | Service | Why | Alternatives and why not |
|---|---|---|---|
| Identity | Identity-Aware Proxy + SSO (Google Workspace or Okta via SAML/OIDC) | Executives log in with the corporate account; the frontend passes a JWT signed by the identity provider whose claims carry the role and brands (from directory groups), and the agent verifies it against the IdP's JWKS. No passwords in the app | App-level OAuth: more code and more to get wrong. API keys per user: no SSO, no offboarding |
| Edge | HTTPS Load Balancer + Cloud Armor | TLS, WAF rules, rate limiting per client | API Gateway: fine for pure APIs, weaker for the web app and websockets/streaming |
| Agent compute | Cloud Run (min 1 instance, autoscale, concurrency ~10) | Stateless container, one instance covers the peak of about 3-12 busy sessions and it scales out if usage grows, streaming responses, pay per use, VPC connector | GKE: operational overhead not justified at a few hundred users. Cloud Functions: request time limits and cold starts for long multi-step turns |
| Background jobs | Cloud Run Jobs / Cloud Run services subscribed to Pub/Sub | Same container image and code as the agent, retries and dead-letter topics from Pub/Sub | Cloud Composer: too heavy for a handful of event handlers |
| LLM | Gemini Flash (main) and Flash-Lite (fallback, guard, extraction, summaries, judge) on Vertex AI | Required by the client; Vertex gives enterprise quotas, SLA, data residency, VPC-SC, no training on our data. Flash-Lite for cheap high-volume calls keeps the main model's quota for reasoning | AI Studio API (used in the prototype): free tier, no SLA, low quotas. Other providers: possible behind the same interface, but the client tests with Gemini |
| Embeddings and retrieval | Vertex AI embeddings + Vertex AI Vector Search (hybrid: dense + sparse/keyword) | Managed ANN index with metadata filters (status, schema version, tables), hybrid search for metric names and SQL terms | BigQuery Vector Search: good for batch, higher latency per query. AlloyDB pgvector: one more database to run. At a few thousand trios any of them works; Vector Search keeps it serverless |
| Analytical data | BigQuery, read-only | The dataset is already there; row access policies by brand, policy tags on PII columns, authorized views, `maximum_bytes_billed`, dry runs, job labels for cost attribution | Copying data to another store: duplication and drift |
| Operational store | Firestore (Native mode) | Serverless, document model fits reports, conversations, checkpoints and preferences; TTL policies for retention; per-document security; no capacity planning | Cloud SQL Postgres: a good choice if reporting over this data in SQL is needed, but one more instance to size, patch and fail over |
| Cache and limits | Memorystore for Redis | Shared rate limiter per model quota across instances, profile cache, query result cache, circuit breaker state | In-process caches (prototype): each instance would have its own limits and caches |
| Files | Cloud Storage | Golden Bucket raw trios (versioned objects), exported reports, future charts; object events start indexing | Storing trios only in the vector DB: no raw history, no easy review |
| Events | Pub/Sub | Decouples the user's turn from learning and indexing; retries, dead letters, ordering keys per user | Cloud Tasks: point-to-point only. Calling workers synchronously: slower turns, failures leak into the user path |
| PII | Cloud DLP | Inspection of free text (answers, saved reports) for PII types regexes miss; de-identification templates | Presidio self-hosted: works, but more to operate |
| Secrets | Secret Manager | API keys (LangSmith, email provider), no secrets in images or env files | Env vars from CI: leak into logs and build history |
| Observability | Cloud Logging, Trace, Monitoring, Error Reporting + LangSmith; log sink to BigQuery | Structured logs with `trace_id`, OpenTelemetry spans, alerting; LangSmith for LLM-level traces, datasets and prompt comparison; BigQuery sink for the failure analysis and dashboards | Langfuse self-hosted: a good alternative if traces must stay in our project; Arize Phoenix similar |
| Persona and prompt versions | Firestore documents + Admin UI (optionally LangSmith prompt hub) | Versioned, reviewable, instant publish without redeploy, rollback | Prompts in code: needs a developer and a deploy |
| CI/CD | GitHub Actions (tests, lint) + Cloud Build (image) + Cloud Deploy (canary) | The eval gate runs before traffic shifts; canary 10% for prompt or model changes | Manual deploys |

---

## 3. Answering a question

```mermaid
sequenceDiagram
    autonumber
    actor U as Executive
    participant C as Client (CLI / Web)
    participant A as Agent service
    participant G as Input guard (rules + Flash-Lite)
    participant R as Golden retrieval
    participant L as Gemini Flash
    participant T as run_sql tool
    participant Q as SQL Guard + ACL
    participant B as BigQuery
    participant M as PII masker
    participant S as Checkpointer

    U->>C: "Why did churn spike last month?"
    C->>A: turn(text, user identity, conversation id)
    A->>S: load conversation state
    A->>M: mask user text before it enters state
    A->>G: classify(latest message + recent turns)
    alt off-topic or malicious
        G-->>A: blocked
        A-->>C: polite refusal (template, no main-model call)
    else analysis / follow-up
        A->>R: retrieve(standalone question)
        R-->>A: top-3 analyst trios (definitions, SQL patterns)
        A->>L: system prompt (core rules, user scope, persona,<br/>preferences, glossary, schema, trios, analysis context) + history
        loop until a final answer, max steps
            L-->>A: tool call run_sql(sql, purpose)
            A->>T: run_sql
            T->>Q: parse, read-only, whitelist, PII projections,<br/>ACL rewrite, LIMIT
            alt rejected
                Q-->>T: rule + how to fix
                T-->>L: ERROR(rejected), attempts left
            else allowed
                T->>B: dry run (free): syntax + bytes
                alt syntax error or too expensive
                    B-->>T: error
                    T-->>L: ERROR(syntax / too_expensive), attempts left
                else ok
                    T->>B: execute (maximum_bytes_billed, timeout)
                    B-->>T: rows
                    T->>M: mask result (column names + regex)
                    alt 0 rows
                        T-->>L: EMPTY, check filter values
                    else rows
                        T-->>L: first N rows + numeric summary
                    end
                end
            end
        end
        L-->>A: final answer
        A->>M: mask answer (regex, DLP in production)
        A->>S: save messages + analysis context
        A-->>C: answer + trace id
        A-)A: async: extract preference signals, summarize old turns
    end
```

Notes:

- **Self-correction is bounded:** after `MAX_SQL_FAILURES` (3) failed attempts in a turn the tool
  returns `STOP` and the model must explain what went wrong. Each turn also has a BigQuery byte
  budget and a step limit, so a confused model cannot loop or run up costs.
- **Dry runs are free** and catch syntax errors, unknown columns and expensive scans before any
  bytes are billed.
- **Follow-ups** ("and only for Jeans?") work because the analysis context (last SQL, period,
  filters, result summary) is in the prompt. The refined SQL goes through the guard and ACL again.
- Multi-step questions ("why", "compare", "report") follow the core prompt rule: state 2 to 4
  hypotheses, run one query per hypothesis, then synthesize with numbers.

---

## 4. Deleting reports with confirmation

```mermaid
sequenceDiagram
    autonumber
    actor U as Executive
    participant C as Client
    participant A as Agent service
    participant L as Gemini
    participant D as request_report_deletion tool
    participant H as Confirmation helper
    participant F as Report store
    participant S as Checkpointer
    participant AU as Audit log

    U->>C: "Delete all reports we made in this conversation"
    C->>A: turn
    A->>L: prompt + history
    L-->>A: tool call request_report_deletion(this_conversation=true)
    A->>D: run
    D->>F: find candidates WHERE owner = current user (deterministic)
    alt no candidates
        D-->>L: NO_MATCH (+ ids not found among the user's reports)
        L-->>C: "No matching reports"
    else candidates
        D->>H: require(action, preview, digest = sha256(id@updated_at))
        H->>AU: confirmation_requested
        H->>S: interrupt: graph paused, state saved
        A-->>C: preview table + "Delete 3 reports? [y/N]"
        Note over L: the model is not called while waiting<br/>and cannot answer for the user
        alt user answers y / clicks Delete
            U->>C: y
            C->>A: resume(answer=yes, digest shown)
            A->>D: re-run the tool call
            D->>F: collect candidates again, compare digest
            alt set changed
                D-->>L: CANCELLED: set changed, ask again
            else same set
                D->>F: delete the reports (hard delete)
                D->>AU: confirmation_approved, reports_deleted
                D-->>L: DELETED: 3 reports
            end
        else no / anything else / new question
            C->>A: resume(answer=no)
            D->>AU: confirmation_rejected
            D-->>L: CANCELLED
            opt the user typed a new question
                C->>A: the question as the next turn
            end
        end
        L-->>C: outcome in plain words
    end
```

What makes it strict:

- **The model can only ask.** The answer to the interrupt comes from the client, never from the
  model. A "yes" typed into the chat history is not a confirmation.
- **Exactly what was shown is deleted.** The digest of the previewed set travels with the answer;
  if a report was added or changed in between, nothing is deleted and the user is asked again.
- **Only the user's own reports**, enforced in the store's queries, not in the prompt. Other
  users' reports and missing ids look the same ("not found"), so their existence is not revealed.
- **UX is not broken:** "no", any other reply, or a new question cancels. A new question is then
  answered right away without retyping.
- The same helper guards every tool with side effects: `export_report` today, e-mail or Slack
  posting tomorrow. A tool registered with `requires_confirmation` that performs its effect
  without an approved confirmation raises an error.

---

## 5. Learning loop

```mermaid
flowchart LR
    subgraph Turn["Every turn"]
        Q[Question] --> ANS[Answer]
        ANS --> FB["👍 / 👎 + comment"]
        ANS --> SIG[Preference signals<br/>reformat, shorter, metrics used]
    end

    subgraph UserLevel["User level"]
        SIG --> PS1[[preference-signals topic]]
        FB --> PS2[[feedback topic]]
        PS1 --> W1[Preference worker<br/>weights, decay, threshold]
        PS2 --> W1
        EXP["Explicit: 'always use tables'<br/>update_user_preference tool"] --> PROF
        W1 --> PROF[(Profile<br/>explicit + inferred)]
        PROF -->|every turn| PROMPT[System prompt<br/>User preferences block]
    end

    subgraph SystemLevel["System level"]
        FB -->|good answer + successful SQL| CAND[[golden-candidates topic]]
        CAND --> QUEUE[(Candidates)]
        QUEUE --> PROMO[Promotion worker<br/>rate limit, SQL Guard under author scope,<br/>dry run + run, dedupe, LLM judge,<br/>method notes, number-leak check]
        PROMO -->|auto-approve| GB[(Golden Bucket GCS)]
        PROMO -->|unsure| QUAR[(Quarantine)]
        QUAR --> AUDIT[Sample audit in Admin UI]
        GB -->|sampled| AUDIT
        AUDIT -->|deprecate| GB
        FB -->|👎 on answers using a trio| DEP[Auto-deprecation<br/>bad from 2+ users and bad > good]
        DEP --> GB
        AN[Analysts write new trios] --> GB
        GB -->|object finalize| IDX[Indexer: embed, dedupe,<br/>metadata, deprecate old]
        IDX --> VS[(Vector index)]
        VS -->|top-k at query time| PROMPT

        LOGS[(Logs and traces<br/>BigQuery sink)] --> FR[Failure report<br/>SQL errors by type, exhausted retries,<br/>guard blocks, 👎, repeated questions]
        FR --> FIX[Prompt / glossary / schema notes / new trios]
        FIX --> EVALG{Automated eval gate<br/>security 100%, pass rate, judge, accuracy<br/>no manual approval}
        PERS[Persona draft] --> PCHK[Static checks] --> EVALG
        EVALG -->|pass| PUB[Publish persona version]
        EVALG -->|pass| DEPLOY[Canary rollout]
        EVALG -->|fail| FIX
        GB -->|approved trios with reference SQL| EVALG
    end
```

- **User level:** explicit preferences apply at once; inferred ones need consistent signals from
  at least two different turns and a confidence above 0.7, and old signals decay (half-life 30
  days). The profile goes into every prompt; the current message overrides it, and it can never
  override safety rules.
- **System level:** good interactions become candidates and an automatic pipeline promotes them
  into the Golden Bucket (no human approval in the loop: it would not scale); people audit only
  what the pipeline was unsure about and a sample of what it approved. Trios whose answers collect
  negative feedback are deprecated automatically. Failures are clustered into a report that drives
  prompt, glossary and trio changes. Every change passes the automated eval gate first; approved
  trios double as eval cases with reference SQL.

---

## 6. Observability and the path of PII

```mermaid
flowchart LR
    U[User input] -->|mask_text| ST[Conversation state]
    BQ[(BigQuery<br/>raw rows with PII)] -->|SQL Guard: no PII in the<br/>final projection| RES[Query result]
    RES -->|mask_dataframe:<br/>PII column names + regex| LLM[Gemini]
    ST --> LLM
    LLM -->|mask_text / DLP| OUT[Answer to user]
    LLM --> RPT[Saved report<br/>masked again]

    subgraph Telemetry["Telemetry: masked before it is written"]
        LOG[JSON event log<br/>trace_id, user, model, tokens,<br/>latency, SQL text, bytes, errors]
        PAY[LLM payload log<br/>full model conversation]
        LS[LangSmith traces<br/>anonymizer = mask_text]
    end

    ST -.-> LOG
    LLM -.-> PAY
    LLM -.-> LS
    RES -.->|row counts only, no data| LOG

    LOG --> SINK[(BigQuery log sink)]
    SINK --> DASH[Dashboards and alerts]
    SINK --> FRP[Failure report]
    LOG --> TR["/trace id, /trace --full, /stats"]
    PAY --> TR
```

Where PII exists, and where it does not:

| Place | PII? | Why |
|---|---|---|
| BigQuery `users` table | Yes | Source data. In production PII columns carry policy tags; the agent's service account has no fine-grained reader role on them, so even `SELECT email` fails at BigQuery |
| SQL text | May contain a literal (`WHERE email = '...'`) | PII may be used in filters and joins (D6). The SQL text is masked before it is logged or traced |
| Query results passed to the model | No | The final projection cannot contain PII columns (SQL Guard), and results are masked by column name and by regex before the model sees them |
| Model answer, saved reports, exports | No | Masked at the agent node (so no checkpoint has unmasked text), again at the output guard; DLP in production |
| User input | Masked | If a user pastes an e-mail, it is masked before it enters the conversation state, the logs or LangSmith |
| Logs, payload log, LangSmith | No | Every string field goes through the masker; LangSmith uses `mask_text` as its anonymizer |

Agent-level metrics (computed by `/stats` in the prototype, Cloud Monitoring in production):
turn success rate and errors by kind; SQL self-correction rate and success of corrections;
latency p50/p95/p99 per turn and per component; tokens and cost per turn, user and day; BigQuery
bytes; guardrail events (input guard blocks by category, SQL Guard rejections by rule, PII
masked, ACL-filtered queries); confirmations requested, approved, rejected and stale; feedback
👍/👎; retries, fallbacks and cooldowns per model. Alerts: error rate above 5% over 15 minutes,
fallback share above 50%, cost per day above budget, any BigQuery access-denied spike.

---

## 7. Prototype vs production

```mermaid
flowchart LR
    subgraph Prototype
        P1[CLI / Streamlit, demo JWT from --user or --token]
        P2[Python process]
        P3[AI Studio Gemini API key]
        P4[SQL Guard ACL rewrite]
        P5[Regex PII masker]
        P6[SQLite app.db:<br/>reports, conversations,<br/>preferences, audit]
        P7[SQLite checkpoints.db]
        P8[golden_trios.json +<br/>numpy embeddings index]
        P9[golden_candidates.jsonl +<br/>promote_candidates.py, audit script]
        P10[In-process rate limiter, caches]
        P11[config/persona.md,<br/>re-read every turn]
        P12[logs/agent.jsonl +<br/>llm_payloads.jsonl + LangSmith]
        P13[Post-turn extraction<br/>in the same process]
        P14[evals/run_eval.py,<br/>failure_report.py, GitHub Actions]
    end
    subgraph Production
        R1[Web / Slack / CLI behind IAP + SSO]
        R2[Cloud Run agent service]
        R3[Vertex AI Gemini]
        R4[SQL Guard + BigQuery row access<br/>policies and policy tags]
        R5[Regex + Cloud DLP]
        R6[Firestore collections]
        R7[Firestore checkpointer]
        R8[GCS Golden Bucket +<br/>Vector Search]
        R9[Pub/Sub candidates +<br/>promotion worker, sample audit]
        R10[Memorystore Redis]
        R11[Persona in Firestore,<br/>versioned, Admin UI]
        R12[Cloud Logging / Trace / Monitoring<br/>+ LangSmith + BigQuery sink]
        R13[Pub/Sub workers]
        R14[Eval job as a CI/CD gate + canary]
    end
    P1 --> R1
    P2 --> R2
    P3 --> R3
    P4 --> R4
    P5 --> R5
    P6 --> R6
    P7 --> R7
    P8 --> R8
    P9 --> R9
    P10 --> R10
    P11 --> R11
    P12 --> R12
    P13 --> R13
    P14 --> R14
```

The code is shaped for this move: storage, checkpointer, embedder, model factory and BigQuery
runner are injected objects, so production replaces implementations without touching the
graph, the tools or the prompt.

---

## 8. Data storage

| Data | Prototype | Production | Who can access | Retention | PII |
|---|---|---|---|---|---|
| Transaction data (`orders`, `order_items`, `products`, `users`) | BigQuery public dataset | BigQuery, read-only; row access policies by brand; policy tags on PII columns | Agent service account, queries run with the user's brand scope | Owned by the data platform | Yes in `users` (names, e-mail, address, postal code, coordinates); never leaves BigQuery unmasked |
| Conversation history and state (messages, tool calls, analysis context, pending confirmations) | `data/checkpoints.db` (LangGraph SqliteSaver) | Firestore checkpointer, encrypted at rest | The owner (checked on resume); on-call via break-glass | 90 days after the last activity (purged at start in the prototype, Firestore TTL in production) | No (masked before saving) |
| Conversation index (owner, title, turns) | `data/app.db` `conversations` | Firestore `conversations` | Owner | Same as history | No |
| Saved reports | `data/app.db` `reports` (question, answer in Markdown, source SQL, entities) | Firestore `reports` + GCS for large artifacts | Owner only (no sharing, D13) | Until the owner deletes them (hard delete, no recovery window) | No (masked on save) |
| Exports | `exports/<id>.md` | GCS bucket with signed URLs | Owner | 30 days | No |
| Audit log (confirmations, deletions, exports) | `data/app.db` `audit_log` | Firestore `audit` + Cloud Audit Logs, exported to BigQuery | Security, on-call (read-only) | 1 year (purged on open in the prototype, Firestore TTL in production) | No |
| User preferences and signals | `data/app.db` `user_preferences`, `preference_signals` | Firestore + Redis cache | Owner (view, change, forget), the agent | Signals decay; forgotten on request or after 12 months inactive | No (whitelisted values only) |
| Golden Bucket trios | `data/golden_trios.json` (15 trios) | GCS (versioned objects) | Analysts write, agent reads | Versioned; deprecated trios kept for audit | No |
| Golden index | `data/golden_index.npz` (rebuilt when trios change) | Vertex AI Vector Search | Agent, indexer | Rebuilt from GCS | No |
| Golden candidates | `data/golden_candidates.jsonl` | Firestore `golden_candidates` | Promotion worker, auditors | 90 days from creation (approved ones live on as trios with method notes only) | No (masked) |
| Persona | `config/persona_draft.md` -> `config/persona.md`, versions in `config/persona_versions/` | Firestore `persona_versions` | Content owners edit, agent reads | All versions kept | No |
| Glossary, PII policy, users and access groups | `config/*.yaml` | Firestore / directory groups | Admins | Versioned | No |
| Schema cache | `data/schema_cache.json` (7 days) | Redis | Agent | 7 days | No (column names only) |
| Query result cache | In-memory, 10 min, 100 entries | Redis, keyed by rewritten SQL (so per access scope) | Agent | 10 min | No (masked results) |
| Event log | `logs/agent.jsonl` | Cloud Logging, sink to BigQuery | On-call, analysts of agent quality | 90 days (they hold masked messages and SQL) | No (masked) |
| Full model conversations | `logs/llm_payloads.jsonl` (`LOG_LLM_PAYLOADS`) | Sampled into Cloud Logging / LangSmith | On-call | 90 days in the prototype; 14 days sampled in production | No (masked) |
| LLM traces | LangSmith (optional) | LangSmith (or self-hosted Langfuse) | On-call, prompt engineers | 14 to 30 days | No (anonymizer) |
| Eval results | `evals/results/<ts>.json` | GCS + BigQuery table per run | Engineering | Kept for comparing versions | No |
| Secrets | `.env` (not committed) | Secret Manager | Service accounts | Rotated | n/a |

---

## 9. Component interaction

### Synchronous calls

| From | To | Protocol | What |
|---|---|---|---|
| Client | Agent service | HTTPS behind IAP; server-sent events for streaming stages | `POST /conversations/{id}/turns {text}` returns stages (`Checking request…`, `Querying BigQuery…`, `Writing answer…`) and the final answer, or a pending confirmation. `POST /conversations/{id}/confirmations {interrupt_id, answer, digest}` resumes |
| Agent | Vertex AI Gemini | gRPC/REST (`generateContent`) | Chat with tools; per-call timeout 60 s, retries only on 429 per-minute and 5xx within a 30 s budget, then fallback |
| Agent | Vertex AI embeddings + Vector Search | REST | Embed the standalone question, `findNeighbors` with filter `status = approved` |
| Agent | BigQuery | REST (jobs.query) | Dry run, then query with `maximum_bytes_billed`, labels `app=retail_agent,user=...`, timeout 60 s |
| Agent | Firestore | gRPC | Checkpoints, reports, preferences, audit (transactions for delete + audit) |
| Agent | Redis | RESP | Rate limiter tokens per model, profile and query caches, circuit state |
| Agent | Cloud DLP | REST | `content.inspect` / `deidentify` on answers and reports |
| Admin UI | Admin API | HTTPS behind IAP | Persona drafts, preview, publish; Golden Bucket audit (quarantine, sampled trios) |

### Asynchronous events (Pub/Sub)

All messages are JSON with `event_id`, `ts`, `trace_id`, `user_id` (internal id, not PII) and a
`schema_version`. Subscriptions are push to Cloud Run with retries and a dead-letter topic;
handlers are idempotent by `event_id`.

| Topic | Producer | Consumer | Payload |
|---|---|---|---|
| `feedback` | Agent (on 👍/👎) | Preference worker; candidate builder; metrics | `{"trace_id", "conversation_id", "user_id", "rating": "good" \| "bad", "comment"}` |
| `preference-signals` | Agent, after the turn | Preference worker | `{"user_id", "signals": [{"key": "detail_level", "value": "brief", "kind": "reformat_request", "weight": 1.0, "evidence": "asked for shorter, previous answer 412 words"}]}` |
| `golden-candidates` | Candidate builder (good feedback + successful SQL) | Candidate store, promotion worker | `{"candidate_id", "question", "sql": [...], "answer_summary", "user_role", "trace_id"}` (masked) |
| `trio-ingested` | GCS object finalize (approved or analyst-written trio) | Indexer job | `{"bucket", "object", "generation"}`; the indexer embeds, deduplicates (cosine > 0.95 with an existing trio goes back to review), writes metadata (tables, metrics, schema version, author, status) |
| `persona-published` | Admin API | Agent instances (invalidate cache) | `{"version", "hash", "published_by"}` |
| `eval-completed` | Eval job | CI/CD, dashboards | `{"run_id", "commit", "models", "summary": {"passed", "total", "security_pass_rate", "judge_mean", "accuracy"}}` |

The user's turn never waits for a Pub/Sub consumer: if the workers are down, answers continue
and learning catches up when they recover.
