# nutrition-assistant-mvp

Streamlit app for generating personalized nutrition plans using OpenAI or Anthropic, with data stored in Supabase Postgres.

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
DATABASE_URL = "postgresql://..."
SUPABASE_URL = "https://<project>.supabase.co"
SUPABASE_SERVICE_KEY = "<service-role-key>"

OPENAI_API_KEY = "sk-..." # optional if using OpenAI
ANTHROPIC_API_KEY = "sk-ant-..." # optional if using Anthropic

[auth]
username = "testuser"
password_hash = "<sha256-of-password>"

# Optional: set true only while debugging
DEBUG_ERRORS = false
```

Generate password hash:

```python
import hashlib
print(hashlib.sha256("your_password".encode()).hexdigest())
```

## Notes for Streamlit Cloud

- Keep API keys preloaded in Secrets if you do not want to type them in the UI.
- Keep `SUPABASE_SERVICE_KEY` private in Secrets only.
- Do not disable CORS/XSRF protection in production.