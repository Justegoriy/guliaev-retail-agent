# Retail Analytics Agent

Conversational agent that answers business questions over the BigQuery public dataset
`bigquery-public-data.thelook_ecommerce`.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill in GOOGLE_API_KEY and GOOGLE_CLOUD_PROJECT
gcloud auth application-default login
python -m retail_agent --user ceo
```

Demo users: `ceo`, `cfo`, `bottoms_manager`, `outerwear_manager` (see `config/users.yaml`).

## Web interface (demo)

```bash
python -m retail_agent.web                  # http://127.0.0.1:8501, localhost only
python -m retail_agent.web --server.port 8600
```

The same agent as the CLI, in a browser. There is no authentication: you pick a demo user in
the sidebar, and each browser tab has its own session for that user. The sidebar also has the
conversation list (new / continue) and the system status (Gemini, BigQuery, model, LangSmith).

- **Chat**: progress stages while the agent works, the trace id (and a LangSmith link when
  tracing is on), 👍 / 👎 with an optional comment. Deleting or exporting reports shows the
  affected reports with **Delete / Cancel** buttons; sending a new message cancels the request.
- **Preferences**: your profile; set a value (same validation as `/prefs set`), forget one key
  or reset all.
- **Reports**: your saved reports as markdown, each with a `.md` download. Reports are
  deleted only through the chat, with confirmation.

Logs, metrics and `python -m retail_agent.observability.inspect` cover web turns the same way
as CLI turns.
