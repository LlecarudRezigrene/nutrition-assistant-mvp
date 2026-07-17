-- Current Supabase schema (public tables)
-- Last updated: 2026-07-17
--
-- MULTI-USER: all four tables have Row Level Security ENABLED with owner-scoped
-- policies (see db/01_multiuser_rls.sql). patients & example_plans carry owner_id
-- (uuid -> auth.users, default auth.uid()); lab_values & diet_plans inherit
-- ownership through their patient. The app connects as the authenticated user
-- (SET LOCAL ROLE authenticated + request.jwt.claims) so RLS filters every query.

CREATE TABLE patients (
  id integer,
  owner_id uuid,  -- FK to auth.users; RLS owner; default auth.uid(); added via db/01_multiuser_rls.sql
  name character varying,
  age integer,
  gender character varying,
  height double precision,
  weight double precision,
  bmi double precision,
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