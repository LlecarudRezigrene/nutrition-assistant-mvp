# Shared code imported by BOTH the Streamlit console (pages/) and the FastAPI
# WhatsApp service (whatsapp_service/). Rules:
#   - Modules here read NO config: no st.secrets, no os.environ, no .env.
#     Every function takes credentials/values as explicit arguments — each
#     side loads its own config (st.secrets vs pydantic-settings).
#   - Must stay compatible with both runtimes (Streamlit Cloud's Python and
#     the local 3.14 service).
