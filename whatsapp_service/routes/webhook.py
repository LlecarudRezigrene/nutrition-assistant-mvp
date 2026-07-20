"""Twilio WhatsApp webhook: POST /webhook/whatsapp.

Slice 1 (walking skeleton): validate → log inbound to whatsapp_messages →
reply with a hardcoded Spanish TwiML message. Claude drafting + the approval
queue replace the hardcoded reply in slice 2.
"""
from fastapi import APIRouter, Request, Response
from twilio.request_validator import RequestValidator
from twilio.twiml.messaging_response import MessagingResponse

from shared.models import Direction, MediaNote, WhatsAppMessage
from whatsapp_service.config import get_settings
from whatsapp_service.services.supabase_svc import log_message

router = APIRouter()

# Hardcoded slice-1 acknowledgement (tú tone). Not the real welcome — that one
# is drafted by Claude and only sent after the nutritionist approves it.
_PLACEHOLDER_REPLY = (
    "¡Hola! 🌱 Gracias por escribirnos. En breve te compartimos la información "
    "para comenzar con tu plan de nutrición."
)


def _public_url(request: Request) -> str:
    """Reconstruct the URL Twilio signed. Behind ngrok the app sees http://
    while Twilio signed the https:// URL — trust X-Forwarded-Proto."""
    url = str(request.url)
    if request.headers.get("x-forwarded-proto") == "https" and url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    return url


def _normalise_number(raw: str) -> str:
    """Twilio sends 'whatsapp:+5215512345678' — store bare E.164.
    Re-add a lost '+' (a literal '+' in form data decodes to a space)."""
    number = raw.removeprefix("whatsapp:").strip()
    if number.isdigit():
        number = "+" + number
    return number


def _media_note(params: dict[str, str]) -> MediaNote | None:
    """Receipt-only note for media messages; content is never downloaded."""
    if int(params.get("NumMedia", "0") or 0) == 0:
        return None
    content_type = params.get("MediaContentType0", "")
    if content_type.startswith("audio/"):
        return MediaNote.VOICE
    if content_type.startswith("image/"):
        return MediaNote.IMAGE
    if content_type == "application/pdf":
        return MediaNote.PDF
    return MediaNote.OTHER


@router.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request) -> Response:
    settings = get_settings()
    form = await request.form()
    params = {key: str(value) for key, value in form.items()}

    if not settings.dev_skip_twilio_signature:
        validator = RequestValidator(settings.twilio_auth_token)
        signature = request.headers.get("X-Twilio-Signature", "")
        if not validator.validate(_public_url(request), params, signature):
            return Response(status_code=403)

    message = WhatsAppMessage(
        owner_id=settings.nutritionist_user_id,
        wa_number=_normalise_number(params.get("From", "")),
        direction=Direction.INBOUND,
        body=params.get("Body") or None,
        media_note=_media_note(params),
        twilio_sid=params.get("MessageSid") or None,
    )
    await log_message(settings, message)

    twiml = MessagingResponse()
    twiml.message(_PLACEHOLDER_REPLY)
    return Response(content=str(twiml), media_type="application/xml")
