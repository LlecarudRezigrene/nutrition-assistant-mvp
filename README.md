# nutrition-assistant-mvp

Streamlit app for generating personalized nutrition plans using OpenAI or Anthropic, with data stored in Supabase Postgres..

## Features

- Single-user login (for testing) using credentials in Streamlit secrets.
- Patient CRUD + lab history stored in PostgreSQL (Supabase).
- AI plan generation and regeneration.
- Optional reference PDF ingestion from Supabase Storage bucket `reference-docs`.

## Requirements

- Python 3.11+
- A PostgreSQL connection string (Supabase)
- Streamlit secrets configured

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
streamlit run streamlit_app.py
```

## Required Streamlit Secrets

Add these keys in `.streamlit/secrets.toml` (local) or Streamlit Cloud Secrets:

```toml
DATABASE_URL = "postgresql://..."   # Supabase transaction pooler (port 6543)
SUPABASE_URL = "https://<project>.supabase.co"
SUPABASE_ANON_KEY = "<anon-public-key>"        # used for user login (Supabase Auth)
SUPABASE_SERVICE_KEY = "<service-role-key>"    # used only for Storage (reference docs)

OPENAI_API_KEY = "sk-..." # optional if using OpenAI
ANTHROPIC_API_KEY = "sk-ant-..." # optional if using Anthropic

# Optional: set true only while debugging
DEBUG_ERRORS = false
```

Authentication uses **Supabase Auth** (email + password). Create nutritionist
accounts in the Supabase dashboard (Authentication → Users) with sign-ups
disabled. Per-user data isolation is enforced by Postgres Row Level Security —
see `db/` for the setup SQL. There is no longer an `[auth]` secret section.

## Notes for Streamlit Cloud

- Keep API keys preloaded in Secrets if you do not want to type them in the UI.
- Keep `SUPABASE_SERVICE_KEY` private in Secrets only.
- Do not disable CORS/XSRF protection in production.
