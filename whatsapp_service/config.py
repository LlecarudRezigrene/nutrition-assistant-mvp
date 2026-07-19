"""Typed configuration for the WhatsApp service (.env via pydantic-settings).

The Streamlit console uses st.secrets; this service uses .env. shared/ code
reads neither — values flow from here as explicit function arguments.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor to this directory so uvicorn's working directory doesn't matter.
_ENV_FILE = Path(__file__).resolve().parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    # Supabase (service role BYPASSES RLS — see shared/supabase_client.py)
    supabase_url: str
    supabase_service_key: str

    # Single-tenant MVP: every service-side write stamps this auth.users uuid
    # as owner_id so rows show up in the nutritionist's console (RLS).
    nutritionist_user_id: str

    # Twilio (empty until the sandbox is set up — slice 2)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_from: str = ""  # e.g. "whatsapp:+14155238886" (sandbox number)
    nutritionist_wa_number: str = ""  # her personal number, E.164; must join the sandbox

    # Claude drafting (slice 2)
    anthropic_api_key: str = ""

    # Intake form link (slice 2+): base URL of the Streamlit app + HMAC secret
    intake_base_url: str = ""  # e.g. "https://<app>.streamlit.app"
    intake_link_secret: str = ""

    # Local dev only: lets curl hit the webhook without a Twilio signature.
    # MUST be false (or absent) anywhere Twilio actually points at.
    dev_skip_twilio_signature: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
