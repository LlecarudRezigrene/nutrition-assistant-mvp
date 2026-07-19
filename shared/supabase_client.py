"""Service-role Supabase client factory.

The service role BYPASSES Row Level Security — it exists for the two writers
that have no logged-in user (the FastAPI webhook and the public intake form).
Anything inserted through this client MUST stamp owner_id explicitly, or the
row will be invisible to every nutritionist in the console.
"""
from supabase import Client, create_client


def create_service_client(url: str, service_key: str) -> Client:
    """Build a service-role client. Credentials come from the caller's own
    config (never read here — see shared/__init__.py)."""
    return create_client(url, service_key)
