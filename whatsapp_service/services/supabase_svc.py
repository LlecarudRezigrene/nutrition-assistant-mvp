"""Service-role Supabase writes for the WhatsApp service.

Everything written here bypasses RLS, so owner_id MUST already be stamped on
the models coming in (see whatsapp_service/CLAUDE.md). The supabase client is
sync — calls are pushed to a thread so route handlers stay async.
"""
import asyncio

from supabase import Client

from shared.models import WhatsAppMessage
from shared.supabase_client import create_service_client
from whatsapp_service.config import Settings

_client_cache: Client | None = None


def _client(settings: Settings) -> Client:
    global _client_cache
    if _client_cache is None:
        _client_cache = create_service_client(
            settings.supabase_url, settings.supabase_service_key
        )
    return _client_cache


async def log_message(settings: Settings, message: WhatsAppMessage) -> None:
    """Insert one row into whatsapp_messages. Raises on failure — the route
    returns 500 and Twilio will retry, which is what we want for a log miss."""
    client = _client(settings)
    record = message.model_dump(mode="json", exclude_none=True, exclude={"id", "created_at"})
    await asyncio.to_thread(
        lambda: client.table("whatsapp_messages").insert(record).execute()
    )
