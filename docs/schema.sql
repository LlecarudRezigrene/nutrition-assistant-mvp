-- Current Supabase schema (public tables)
-- Last updated: 2026-07-13

CREATE TABLE patients (
  id integer,
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
  title character varying,
  patient_profile text,
  plan_content text,
  tags json,
  created_at timestamp without time zone
);