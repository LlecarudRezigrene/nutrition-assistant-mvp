-- ============================================================================
-- Phase 2: Per-nutritionist ownership + Row Level Security (RLS)
-- ============================================================================
-- Run in: Supabase dashboard → SQL Editor → paste → Run.
--
-- PREREQUISITES:
--   • Phase 0 (00_delete_test_data.sql) already run — no test rows remain.
--   • Email auth enabled and your nutritionist account(s) created.
--
-- SAFE TO RUN WHILE THE APP IS LIVE: the app currently connects as a Postgres
-- superuser, which BYPASSES RLS — so enabling RLS here does not break anything.
-- RLS only starts enforcing once the app is switched to connect as the
-- logged-in user (the Phase 3 code change).
--
-- Idempotent: re-running is safe (IF NOT EXISTS / DROP POLICY IF EXISTS).
-- ============================================================================

-- 1) Ownership columns → reference the Supabase Auth user (auth.users.id is uuid).
--    ON DELETE CASCADE: removing a nutritionist account removes their data.
ALTER TABLE patients      ADD COLUMN IF NOT EXISTS owner_id uuid REFERENCES auth.users(id) ON DELETE CASCADE;
ALTER TABLE example_plans ADD COLUMN IF NOT EXISTS owner_id uuid REFERENCES auth.users(id) ON DELETE CASCADE;

-- Backstop: auto-fill owner from the caller's JWT on insert. The app also sets
-- it explicitly, but this guarantees a row can never be created ownerless.
ALTER TABLE patients      ALTER COLUMN owner_id SET DEFAULT auth.uid();
ALTER TABLE example_plans ALTER COLUMN owner_id SET DEFAULT auth.uid();

-- Index the owner columns (every query filters by owner).
CREATE INDEX IF NOT EXISTS idx_patients_owner      ON patients (owner_id);
CREATE INDEX IF NOT EXISTS idx_example_plans_owner ON example_plans (owner_id);

-- 2) Privileges. After cutover the app connects as the `authenticated` role.
--    Grant table + sequence access; RLS below restricts WHICH rows it may touch.
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON patients, lab_values, diet_plans, example_plans TO authenticated;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO authenticated;

-- 3) Turn on Row Level Security for every table holding patient data.
ALTER TABLE patients      ENABLE ROW LEVEL SECURITY;
ALTER TABLE lab_values    ENABLE ROW LEVEL SECURITY;
ALTER TABLE diet_plans    ENABLE ROW LEVEL SECURITY;
ALTER TABLE example_plans ENABLE ROW LEVEL SECURITY;

-- 4) Policies. patients & example_plans are owned directly by owner_id.
DROP POLICY IF EXISTS patients_owner ON patients;
CREATE POLICY patients_owner ON patients
  FOR ALL TO authenticated
  USING (owner_id = auth.uid())
  WITH CHECK (owner_id = auth.uid());

DROP POLICY IF EXISTS example_plans_owner ON example_plans;
CREATE POLICY example_plans_owner ON example_plans
  FOR ALL TO authenticated
  USING (owner_id = auth.uid())
  WITH CHECK (owner_id = auth.uid());

-- lab_values & diet_plans have no owner_id — ownership flows through their
-- parent patient, so the policy checks the patient's owner.
DROP POLICY IF EXISTS lab_values_owner ON lab_values;
CREATE POLICY lab_values_owner ON lab_values
  FOR ALL TO authenticated
  USING      (EXISTS (SELECT 1 FROM patients p WHERE p.id = lab_values.patient_id AND p.owner_id = auth.uid()))
  WITH CHECK (EXISTS (SELECT 1 FROM patients p WHERE p.id = lab_values.patient_id AND p.owner_id = auth.uid()));

DROP POLICY IF EXISTS diet_plans_owner ON diet_plans;
CREATE POLICY diet_plans_owner ON diet_plans
  FOR ALL TO authenticated
  USING      (EXISTS (SELECT 1 FROM patients p WHERE p.id = diet_plans.patient_id AND p.owner_id = auth.uid()))
  WITH CHECK (EXISTS (SELECT 1 FROM patients p WHERE p.id = diet_plans.patient_id AND p.owner_id = auth.uid()));

-- After this runs, the database is ready. RLS goes LIVE when the app cuts over
-- to connecting as the authenticated user (Phase 3, code change).
