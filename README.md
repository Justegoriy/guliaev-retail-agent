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

Demo users: `ceo`, `cfo`, `head_men`, `head_women` (see `config/users.yaml`).
