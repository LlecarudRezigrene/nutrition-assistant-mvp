-- ============================================================================
-- Harden the `authenticated` role's table privileges.
-- ============================================================================
-- The app only needs row-level CRUD. Extra privileges appeared from Supabase
-- defaults / the policy template:
--   • TRUNCATE  — ignores RLS entirely (a way to wipe a whole table); remove it.
--   • TRIGGER / REFERENCES — not used by the app.
-- CRUD (SELECT/INSERT/UPDATE/DELETE) stays; RLS restricts it to the user's rows.
-- Safe to run anytime; idempotent (REVOKE of an absent privilege is a no-op).
-- ============================================================================
REVOKE TRUNCATE, TRIGGER, REFERENCES
  ON patients, lab_values, diet_plans, example_plans
  FROM authenticated;
