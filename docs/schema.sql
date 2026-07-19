-- Current Supabase schema (public tables)
-- Last updated: 2026-07-18
--
-- MULTI-USER: all tables have Row Level Security ENABLED with owner-scoped
-- policies (see db/01_multiuser_rls.sql, db/04_whatsapp.sql). patients &
-- example_plans carry owner_id (uuid -> auth.users, default auth.uid());
-- lab_values & diet_plans inherit ownership through their patient. The app
-- connects as the authenticated user (SET LOCAL ROLE authenticated +
-- request.jwt.claims) so RLS filters every query.
--
-- WHATSAPP INTAKE (db/04_whatsapp.sql): whatsapp_messages, pending_approvals,
-- intake_submissions are written by the FastAPI service / public intake form
-- via the SERVICE ROLE (bypasses RLS, stamps owner_id explicitly) and read by
-- the console through RLS.

CREATE TABLE patients (
  id integer,
  owner_id uuid,  -- FK to auth.users; RLS owner; default auth.uid(); added via db/01_multiuser_rls.sql
  name character varying,
  age integer,
  gender character varying,
  height double precision,
  weight double precision,
  bmi double precision,
  whatsapp_number character varying,  -- E.164 (+5215512345678), UNIQUE; links WhatsApp senders to records; added via db/04_whatsapp.sql
  health_conditions json,
  nutrient_targets json,  -- per-patient daily nutrient targets; added via init_db ALTER
  created_at timestamp without time zone,
  updated_at timestamp without time zone
);

CREATE TABLE lab_values (
  id integer,
  patient_id integer,  -- FK to patients.id
  test_date character varying,
  glucose double precision,
  cholesterol double precision,
  triglycerides double precision,
  hemoglobin double precision,
  created_at timestamp without time zone
);

CREATE TABLE diet_plans (
  id integer,
  patient_id integer,  -- FK to patients.id
  plan_details text,
  special_considerations text,
  status character varying,
  created_at timestamp without time zone,
  updated_at timestamp without time zone
);

CREATE TABLE example_plans (
  id integer,
  owner_id uuid,  -- FK to auth.users; RLS owner (private per nutritionist); added via db/01_multiuser_rls.sql
  title character varying,
  patient_profile text,
  plan_content text,
  tags json,
  created_at timestamp without time zone
);

-- ── WhatsApp intake tables (db/04_whatsapp.sql) ─────────────────────────────

CREATE TABLE whatsapp_messages (
  id bigint,           -- bigserial PK
  owner_id uuid,       -- FK to auth.users; RLS owner; service writes stamp it explicitly
  wa_number character varying,   -- patient's number, E.164
  direction character varying,   -- 'inbound' | 'outbound'
  body text,
  media_note character varying,  -- 'voice'|'image'|'pdf'|'other' — receipt only, content never downloaded
  twilio_sid character varying,
  patient_id integer,  -- FK to patients.id, ON DELETE SET NULL
  created_at timestamp with time zone
);

CREATE TABLE pending_approvals (
  id bigint,
  owner_id uuid,
  wa_number character varying,
  draft_text text,               -- Claude's draft, as generated
  approved_text text,            -- what was actually sent (after edits)
  status character varying,      -- 'pending' | 'sent' | 'rejected' | 'send_failed'
  created_at timestamp with time zone,
  resolved_at timestamp with time zone
);

CREATE TABLE intake_submissions (
  id bigint,
  owner_id uuid,
  wa_number character varying,
  form_data jsonb,               -- raw form answers, kept as audit trail
  consent_accepted_at timestamp with time zone,  -- LFPDPPP express consent
  patient_id integer,            -- FK to patients.id, ON DELETE SET NULL
  status character varying,      -- 'new' | 'patient_created' | 'duplicate'
  seen_by_nutritionist boolean,
  created_at timestamp with time zone
);