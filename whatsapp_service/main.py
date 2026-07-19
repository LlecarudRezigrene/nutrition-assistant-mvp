"""FastAPI app for the WhatsApp intake service.

Run from the REPO ROOT (so `shared/` imports resolve):
    uvicorn whatsapp_service.main:app --reload --port 8000
"""
from fastapi import FastAPI

from whatsapp_service.routes.webhook import router as webhook_router

# No /docs or /redoc: this service is exposed through ngrok — don't advertise
# the API surface to whoever finds the URL.
app = FastAPI(title="WhatsApp intake service", docs_url=None, redoc_url=None)
app.include_router(webhook_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
