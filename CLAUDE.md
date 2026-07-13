# CLAUDE.md

Spanish-language AI nutrition assistant for a nutritionist in Mexico. A logged-in user manages patients and lab values, then generates personalized diet plans with OpenAI or Anthropic, editable and downloadable as Word documents.

## Tech stack

- **Streamlit** single-file app: everything lives in `streamlit_app.py` (~1,400 lines). Deployed on Streamlit Cloud; config/secrets via `st.secrets`.
- **SQLAlchemy + psycopg2** over **Supabase Postgres** (`DATABASE_URL`). Tables auto-created with `Base.metadata.create_all()` — no migrations; `create_all` never alters existing tables.
- **supabase** client only for Storage: reference PDFs in the `reference-docs` bucket, injected into the AI prompt.
- **openai** / **anthropic** SDKs — provider chosen at runtime in the sidebar. Model names are module constants (`OPENAI_MODEL`, `ANTHROPIC_MODEL`) near the "AI generation" section.
- **pandas + altair** for lab trend charts (altair ships with Streamlit; it's not in requirements.txt).
- **python-docx / PyPDF2** for Word export and PDF/docx text extraction.

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Requires `.streamlit/secrets.toml` — see README for the full list (`DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, optional API keys, `[auth]` username + SHA-256 password_hash).

Theme lives in `.streamlit/config.toml` (committed; slate + teal, `#0D9488` primary). Global CSS (`_APP_CSS` after page config) sets the Inter font and hides Streamlit chrome — it keeps `stHeader` visible-but-transparent and pads the page 3.75rem to clear its fixed height; don't reduce that padding or content clips under the header. Secrets stay gitignored.

## File layout (single file, banner-comment sections)

`streamlit_app.py` top-to-bottom: page config → auth (gates everything via `st.stop()`) → clinical constants (`LAB_METRICS`, BMI) → SQLAlchemy models → DB/Supabase init → helpers → example-plan matching → prompt builder → AI generation → DOCX export → UI render functions → main UI (sidebar, patient selector, four tabs). Keep new code in the matching section.

Quirk: `with tab_generar:` is entered **twice** (generation UI, then later the current-plan/regenerate UI after the history tab) — intentional, don't "fix" it.

## Database tables

Defined as SQLAlchemy models; documented in `docs/schema.sql`. Note: that file may lag the live DB — verify constraints/types against Supabase before relying on it (known open question: live `created_at`/`updated_at` may be `timestamp without time zone` while models say `timezone=True`).

- **patients**: id PK, name, age, gender (`male`/`female`/`other` — UI maps via `GENDER_TO_DB`), weight, height, bmi, health_conditions (json list), created_at, updated_at
  - `health_conditions` is entered via a structured picker (`CONDITION_OPTIONS` multiselect + CKD stage/dialysis follow-up + free-text "other"), composed into the flat list by `_compose_conditions` and parsed back by `_decompose_conditions`. No schema change — CKD detail is stored as a string like `"Enfermedad renal crónica etapa G4, sin diálisis"`. Legacy free-text conditions decompose into the "other" field.
- **lab_values**: id PK, patient_id FK→patients (CASCADE), test_date (varchar, ISO `YYYY-MM-DD` — sorts as string), glucose, cholesterol, triglycerides, hemoglobin, created_at
- **diet_plans**: id PK, patient_id FK→patients (CASCADE), plan_details (text, markdown), special_considerations, status (default `active`), created_at, updated_at
- **example_plans**: id PK, title, patient_profile, plan_content, tags (json list), created_at

## Conventions

- **Spanish UI, English code.** All identifiers/comments/docstrings in English; every user-facing string in Spanish with emoji prefixes (✅ ❌ ⚠️ 🚨).
- **Small `_`-prefixed helpers** for anything reused (`_sanitise`, `_gs`, `_parse_csv`, `_show_error`). Factor even 1–2-line utilities.
- **Config as module constants**: `_STATE_DEFAULTS` drives session-state init AND `reset_form()` — add new session keys there, not inline. `LAB_METRICS` drives status logic, charts, and summary cards.
- **Comments explain why, not what** (e.g. why `delta_color="off"` is clinically correct).

## Architectural rules

- All DB access goes through the `get_db()` context manager (commit/rollback/close). Call `session.expunge_all()` before using ORM objects after the session closes.
- Every DB/AI/file operation is wrapped in try/except with a Spanish `st.error` via `_show_error` — the app must never crash on the user. Errors are generic to the user; exception detail only shows with `DEBUG_ERRORS = true` in secrets. DB connection failure fails closed with `st.stop()`.
- Anything user-entered that goes into `unsafe_allow_html` markup must pass through `html.escape()` (see `render_patient_summary`).
- Login throttle state lives in `@st.cache_resource` (cross-session), NOT session state — session state resets on refresh, which would defeat the lockout.
- Extracted document text (uploads, reference PDFs) is capped at `MAX_DOC_CHARS`.
- Streamlit idioms: unique widget `key=` with prefixes for repeated components (`f"{prefix}_dl_{plan.id}"`), `st.rerun()` after state mutations, `@st.cache_resource` for clients/engine, `@st.cache_data(ttl=...)` for fetched data.
- Destructive actions use a two-step confirm (pending flag in session state — see regenerate flow and plan delete in `render_plan_card`).
- Admin flows (example plans) live in `@st.dialog` modals, not inline in the main page. Inside a dialog, `st.rerun()` closes it; `st.rerun(scope="fragment")` refreshes it in place.
- Every tab renders an empty-state `st.info` when no patient is selected — never a blank tab.
- The delivered plan must NOT restate the patient's diagnoses: `build_plan_docx` omits conditions from the header, and the prompt instructs the model to use conditions only for clinical reasoning, never naming them in the output.
- Nutritionist can set optional daily nutrient targets (`NUTRIENT_TARGETS`) in the Generar tab — number inputs keyed `target_<key>`, read via `_collect_targets()`, formatted by `_format_targets()`. A set target (non-zero) takes priority over guideline defaults in the prompt.
- Prompt is split for caching: `_build_reference_system()` returns the role + reference documents (identical across patients) and is sent as a **cached** system prompt (`cache_system=True` in `_ai_complete` → Anthropic `cache_control`; OpenAI caches long prefixes automatically). `_build_patient_prompt()` returns the patient-specific user message (data + selected example plans + output spec). Reference docs are capped at `MAX_DOC_CHARS` (120k — large because they're cached once, not re-billed per patient).
- Untrusted text going into LLM prompts passes through `_sanitise()` (strips non-printables, truncates). Reference docs (system) and example plans (user) are wrapped in INICIO/FIN delimiters and the prompt instructs the model to ignore instructions inside them. Credentials compare via `hmac.compare_digest`. Keep all three habits.
- User-facing datetimes format as `%d/%m/%Y %H:%M`.
