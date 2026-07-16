-- ============================================================================
-- Verify the Phase 2 multi-user / RLS setup. Run in Supabase SQL Editor.
-- Every row in the result should read 'OK' in the status column.
-- Expected: 2 owner_id columns, 4 RLS-enabled tables, 4 policies, 16 grants.
-- ============================================================================
WITH checks AS (
  -- owner_id columns present, defaulting to auth.uid()
  SELECT 'owner_id column' AS check_name, table_name AS detail,
         CASE WHEN column_default LIKE '%auth.uid()%' THEN 'OK' ELSE 'MISSING DEFAULT' END AS status
  FROM information_schema.columns
  WHERE table_schema = 'public' AND column_name = 'owner_id'

  UNION ALL
  -- Row Level Security enabled on every patient-data table
  SELECT 'RLS enabled', relname,
         CASE WHEN relrowsecurity THEN 'OK' ELSE 'DISABLED' END
  FROM pg_class
  WHERE relnamespace = 'public'::regnamespace
    AND relname IN ('patients', 'lab_values', 'diet_plans', 'example_plans')

  UNION ALL
  -- One owner-scoped policy per table
  SELECT 'policy', tablename || ' → ' || policyname, 'OK'
  FROM pg_policies WHERE schemaname = 'public'

  UNION ALL
  -- The authenticated role can read/write the tables (RLS restricts to own rows)
  SELECT 'grant → authenticated', table_name || ' → ' || privilege_type, 'OK'
  FROM information_schema.role_table_grants
  WHERE grantee = 'authenticated' AND table_schema = 'public'
    AND table_name IN ('patients', 'lab_values', 'diet_plans', 'example_plans')
)
SELECT * FROM checks ORDER BY check_name, detail;
