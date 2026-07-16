-- ============================================================================
-- Phase 0: Delete ALL test patient data before real patients are entered.
-- ============================================================================
-- ⚠️  IRREVERSIBLE. Run this ONLY while the database still contains test data
--     and NO real patient has been entered yet. Verify first with the SELECTs.
--
-- Run in: Supabase dashboard → SQL Editor → paste → Run.
-- Reference guideline documents live in Storage (the reference-docs bucket),
-- NOT in these tables, so they are unaffected.
-- ============================================================================

-- 1) Look before you delete — how many rows are about to go?
SELECT
  (SELECT count(*) FROM patients)      AS patients,
  (SELECT count(*) FROM lab_values)    AS lab_values,
  (SELECT count(*) FROM diet_plans)    AS diet_plans,
  (SELECT count(*) FROM example_plans) AS example_plans;

-- 2) Delete patient data. lab_values and diet_plans cascade from patients via
--    their ON DELETE CASCADE foreign keys, but we delete explicitly for clarity.
DELETE FROM lab_values;
DELETE FROM diet_plans;
DELETE FROM patients;

-- 3) Example plans are test data too — clear them (reference guideline docs in
--    Storage are NOT affected). New per-nutritionist examples start fresh.
DELETE FROM example_plans;

-- 4) Confirm the tables are empty (expect 0 for the ones you cleared).
SELECT
  (SELECT count(*) FROM patients)      AS patients,
  (SELECT count(*) FROM lab_values)    AS lab_values,
  (SELECT count(*) FROM diet_plans)    AS diet_plans,
  (SELECT count(*) FROM example_plans) AS example_plans;
