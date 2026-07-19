-- ============================================================================
-- WhatsApp intake MVP: patients.whatsapp_number + three new tables.
-- ============================================================================
-- Run in: Supabase dashboard → SQL Editor → paste → Run.
--
-- PREREQUISITES: db/01 (RLS) and db/03 (grant hardening) already run.
--
-- Ownership model: the FastAPI webhook service and the public intake form have
-- no logged-in user, so they write via the SERVICE ROLE (bypasses RLS) and
-- stamp owner_id = the nutritionist's auth uuid explicitly. The Streamlit
-- console reads these tables through the normal RLS path, so each nutritionist
-- only ever sees their own queue. DEFAULT auth.uid() is a backstop for
-- console-side inserts, same as patients/example_plans.
--
-- Idempotent: re-running is safe (IF NOT EXISTS / DROP POLICY IF EXISTS).
-- ============================================================================

-- 1) Link column: THE identifier tying WhatsApp senders to patient records.
--    E.164 format, e.g. +5215512345678. NULL for console-created patients
--    until the nutritionist adds a number. UNIQUE so one number = one patient.
ALTER TABLE patients ADD COLUMN IF NOT EXISTS whatsapp_number varchar UNIQUE;

-- 2) Every inbound and outbound WhatsApp message.
CREATE TABLE IF NOT EXISTS whatsapp_messages (
  id          bigserial PRIMARY KEY,
  owner_id    uuid NOT NULL DEFAULT auth.uid() REFERENCES auth.users(id) ON DELETE CASCADE,
  wa_number   varchar NOT NULL,                 -- patient's number, E.164
  direction   varchar NOT NULL CHECK (direction IN ('inbound','outbound')),
  body        text,
  media_note  varchar,                          -- 'voice'|'image'|'pdf'|'other' — receipt logged, content never downloaded
  twilio_sid  varchar,                          -- Twilio MessageSid for tracing
  patient_id  integer REFERENCES patients(id) ON DELETE SET NULL,
  created_at  timestamptz NOT NULL DEFAULT now()
);

-- 3) Outbound drafts awaiting nutritionist approval. NOTHING is sent to a
--    patient without a row here reaching status 'sent' via the console.
CREATE TABLE IF NOT EXISTS pending_approvals (
  id            bigserial PRIMARY KEY,
  owner_id      uuid NOT NULL DEFAULT auth.uid() REFERENCES auth.users(id) ON DELETE CASCADE,
  wa_number     varchar NOT NULL,
  draft_text    text NOT NULL,                  -- Claude's draft, as generated
  approved_text text,                           -- what was actually sent (after any edits)
  status        varchar NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','sent','rejected','send_failed')),
  created_at    timestamptz NOT NULL DEFAULT now(),
  resolved_at   timestamptz
);

-- 4) Raw intake-form submissions, kept even after a patients row is created
--    (audit trail + LFPDPPP consent evidence).
CREATE TABLE IF NOT EXISTS intake_submissions (
  id                   bigserial PRIMARY KEY,
  owner_id             uuid NOT NULL DEFAULT auth.uid() REFERENCES auth.users(id) ON DELETE CASCADE,
  wa_number            varchar NOT NULL,
  form_data            jsonb NOT NULL,          -- raw answers as submitted
  consent_accepted_at  timestamptz NOT NULL,    -- express consent timestamp (datos sensibles)
  patient_id           integer REFERENCES patients(id) ON DELETE SET NULL,
  status               varchar NOT NULL DEFAULT 'new'
                       CHECK (status IN ('new','patient_created','duplicate')),
  seen_by_nutritionist boolean NOT NULL DEFAULT false,  -- drives the "Nuevo paciente" notice
  created_at           timestamptz NOT NULL DEFAULT now()
);

-- 5) Indexes: console lists filter by owner; webhook matches by number.
CREATE INDEX IF NOT EXISTS idx_wa_messages_owner        ON whatsapp_messages (owner_id);
CREATE INDEX IF NOT EXISTS idx_wa_messages_number       ON whatsapp_messages (wa_number);
CREATE INDEX IF NOT EXISTS idx_pending_approvals_owner  ON pending_approvals (owner_id);
CREATE INDEX IF NOT EXISTS idx_intake_submissions_owner ON intake_submissions (owner_id);

-- 6) Privileges for the console's `authenticated` role: row-level CRUD only,
--    same shape as db/01 + the db/03 hardening (no TRUNCATE/TRIGGER/REFERENCES).
GRANT SELECT, INSERT, UPDATE, DELETE ON whatsapp_messages, pending_approvals, intake_submissions TO authenticated;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO authenticated;
REVOKE TRUNCATE, TRIGGER, REFERENCES ON whatsapp_messages, pending_approvals, intake_submissions FROM authenticated;

-- 7) Row Level Security: owner-scoped, mirroring patients/example_plans.
ALTER TABLE whatsapp_messages  ENABLE ROW LEVEL SECURITY;
ALTER TABLE pending_approvals  ENABLE ROW LEVEL SECURITY;
ALTER TABLE intake_submissions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS whatsapp_messages_owner ON whatsapp_messages;
CREATE POLICY whatsapp_messages_owner ON whatsapp_messages
  FOR ALL TO authenticated
  USING (owner_id = auth.uid())
  WITH CHECK (owner_id = auth.uid());

DROP POLICY IF EXISTS pending_approvals_owner ON pending_approvals;
CREATE POLICY pending_approvals_owner ON pending_approvals
  FOR ALL TO authenticated
  USING (owner_id = auth.uid())
  WITH CHECK (owner_id = auth.uid());

DROP POLICY IF EXISTS intake_submissions_owner ON intake_submissions;
CREATE POLICY intake_submissions_owner ON intake_submissions
  FOR ALL TO authenticated
  USING (owner_id = auth.uid())
  WITH CHECK (owner_id = auth.uid());

-- ============================================================================
-- Verification (run after): expect 3 tables with rowsecurity = true and one
-- owner policy each, and patients.whatsapp_number present.
-- ============================================================================
-- SELECT tablename, rowsecurity FROM pg_tables
--   WHERE tablename IN ('whatsapp_messages','pending_approvals','intake_submissions');
-- SELECT tablename, policyname FROM pg_policies
--   WHERE tablename IN ('whatsapp_messages','pending_approvals','intake_submissions');
-- SELECT column_name FROM information_schema.columns
--   WHERE table_name = 'patients' AND column_name = 'whatsapp_number';
