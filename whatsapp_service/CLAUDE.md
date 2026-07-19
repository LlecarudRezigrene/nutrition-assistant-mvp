# whatsapp_service/CLAUDE.md

FastAPI service handling the WhatsApp side of patient intake: Twilio webhook in,
message logging, Claude draft generation into the approval queue, notifications
to the nutritionist. Runs separately from the Streamlit console (which approves
and sends). See the root CLAUDE.md "WhatsApp intake" section for the monorepo
picture; rules here are service-specific.

## Run (local dev)

```bash
# From the REPO ROOT (so `shared/` imports resolve; venv lives in whatsapp_service/.venv)
whatsapp_service/.venv/Scripts/uvicorn whatsapp_service.main:app --reload --port 8000

# Expose to Twilio:
ngrok http 8000
# Then paste https://<id>.ngrok-free.app/webhook/whatsapp into the Twilio
# sandbox "When a message comes in" box — again every time ngrok restarts.
```

Config is `whatsapp_service/.env` (copy `.env.example`), typed by `config.py`
(pydantic-settings). Python 3.14 venv: `py -3.14 -m venv whatsapp_service/.venv`.

## Style

- Type hints on **every** function; `async def` for all I/O paths (sync SDK
  calls go through `asyncio.to_thread` — see `services/supabase_svc.py`).
- Pydantic models (in `shared/models.py`) for every request/response and DB row.
- English code/comments/logs; Spanish (tú tone) for anything a patient sees.

## Twilio rules

- Validate `X-Twilio-Signature` on every webhook request; invalid → 403 with no
  body. `DEV_SKIP_TWILIO_SIGNATURE=true` is for local curl only — never where
  Twilio actually points.
- Reconstruct the signed URL via `X-Forwarded-Proto` (ngrok terminates TLS —
  see `_public_url` in `routes/webhook.py`).
- Answer TwiML fast (Twilio times out ~15 s): log, queue, return. Anything slow
  (Claude calls) must not block the 200.
- Media messages (voice/image/pdf): log a `media_note` receipt only. Never
  download media content.
- 24-h session rule: freeform sends only work within 24 h of the recipient's
  last inbound message (sandbox AND production). Send failures set the approval
  row to `send_failed`, never crash.

## Claude prompts

- Prompts live in `prompts/*.md` — English instructions, Spanish-tú output —
  loaded at call time so iterating needs no code change.
- Patient message text is UNTRUSTED: sanitise and wrap in INICIO/FIN delimiters
  before it enters a prompt, and instruct the model to ignore instructions
  inside them (same habit as the main app's `_sanitise`).
- Drafts are NEVER sent from this service to a patient. They go to
  `pending_approvals`; only the console's Aprobar button sends.

## Data rules

- The service-role client bypasses RLS → **every insert stamps
  `owner_id = settings.nutritionist_user_id`**. An unstamped row is invisible
  in the console and orphaned.
- Normalise phone numbers to bare E.164 (`+5215512345678`) at the boundary
  (strip the `whatsapp:` prefix on arrival, add it back only when sending).
- Message bodies are sensitive health data (LFPDPPP): store them in the DB,
  never print them to stdout/logs.
- `.env` holds the god-key (`SUPABASE_SERVICE_KEY`); it stays gitignored and
  never appears in code, logs, or error messages.
