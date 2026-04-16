import streamlit as st
import io
import hashlib
import hmac
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime, Text, ForeignKey, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship
from datetime import datetime, timezone
import pandas as pd
from openai import OpenAI
import anthropic
from supabase import create_client as create_supabase_client

# ──────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────---
st.set_page_config(page_title="Asistente de Nutrición con IA", page_icon="🥗", layout="wide")

# ──────────────────────────────────────────────
# Authentication
# ──────────────────────────────────────────────
def _check_credentials(username: str, password: str) -> bool:
    try:
        stored_user = st.secrets["auth"]["username"]
        stored_hash = st.secrets["auth"]["password_hash"]
    except KeyError:
        st.error("❌ Credenciales no configuradas en secrets.")
        return False
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return hmac.compare_digest(username, stored_user) and hmac.compare_digest(password_hash, stored_hash)


def login_page():
    if st.session_state.get("authenticated"):
        return True
    st.title("🥗 Asistente de Nutrición con IA")
    st.markdown("---")
    st.subheader("🔐 Iniciar Sesión")
    with st.form("login_form"):
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Entrar", type="primary")
    if submitted:
        if _check_credentials(username, password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Usuario o contraseña incorrectos")
    return False


if not login_page():
    st.stop()

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
GENDER_TO_DB = {"Masculino": "male", "Femenino": "female", "Otro": "other"}
GENDER_FROM_DB = {v: k for k, v in GENDER_TO_DB.items()}
GENDER_OPTIONS = list(GENDER_TO_DB.keys())

AGE_KEYWORDS = [(18, ["adolescente", "joven"]), (30, ["adulto joven"]), (60, ["adulto"]), (999, ["adulto mayor", "tercera edad"])]
BMI_KEYWORDS = [(18.5, "bajo peso"), (25, "peso normal"), (30, "sobrepeso"), (999, "obesidad")]

LAB_METRICS = {
    "Glucosa (mg/dL)": {"normal": (70, 100), "color": "#FF6B6B"},
    "Colesterol (mg/dL)": {"normal": (0, 200), "color": "#4ECDC4"},
    "Triglicéridos (mg/dL)": {"normal": (0, 150), "color": "#45B7D1"},
    "Hemoglobina (g/dL)": {"normal": (12, 17), "color": "#96CEB4"},
}

REFERENCE_BUCKET = "reference-docs"

_STATE_DEFAULTS = {
    "patient_created": False, "plan_generated": False,
    "current_patient_id": None, "current_plan": None, "current_plan_id": None,
    "load_existing_patient": False, "show_add_example": False,
    "patient_name": "", "patient_age": 30, "patient_gender": "Masculino",
    "patient_weight": 70.0, "patient_height": 170.0, "patient_health_conditions": "",
    "patient_glucose": 0.0, "patient_cholesterol": 0.0, "patient_triglycerides": 0.0, "patient_hemoglobin": 0.0,
}


# ──────────────────────────────────────────────
# Database models
# ──────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


def _utcnow():
    return datetime.now(timezone.utc)


class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    weight = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    health_conditions = Column(JSON, default=list)
    bmi = Column(Float)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    lab_values = relationship("LabValue", back_populates="patient", cascade="all, delete-orphan")
    diet_plans = relationship("DietPlan", back_populates="patient", cascade="all, delete-orphan")


class LabValue(Base):
    __tablename__ = "lab_values"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"))
    test_date = Column(String, nullable=False)
    glucose = Column(Float)
    cholesterol = Column(Float)
    triglycerides = Column(Float)
    hemoglobin = Column(Float)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    patient = relationship("Patient", back_populates="lab_values")


class DietPlan(Base):
    __tablename__ = "diet_plans"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"))
    plan_details = Column(Text, nullable=False)
    special_considerations = Column(Text)
    status = Column(String, default="active")
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    patient = relationship("Patient", back_populates="diet_plans")


class ExamplePlan(Base):
    __tablename__ = "example_plans"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    patient_profile = Column(Text)
    plan_content = Column(Text, nullable=False)
    tags = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), default=_utcnow)


# ──────────────────────────────────────────────
# Database & Supabase init
# ──────────────────────────────────────────────
@st.cache_resource
def init_db():
    if "DATABASE_URL" not in st.secrets:
        st.error("❌ DATABASE_URL no está configurado en secrets.")
        st.stop()
    database_url = st.secrets["DATABASE_URL"]
    if not database_url.startswith("postgresql://"):
        st.error("❌ DATABASE_URL debe comenzar con 'postgresql://'")
        st.stop()
    try:
        engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=3600)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        Base.metadata.create_all(engine)
        return sessionmaker(bind=engine)
    except Exception:
        st.error("❌ No se pudo conectar con la base de datos.")
        st.stop()


SessionFactory = init_db()


@contextmanager
def get_db():
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@st.cache_resource
def get_supabase_client():
    try:
        return create_supabase_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_SERVICE_KEY"])
    except KeyError:
        return None


@st.cache_data(ttl=3600)
def load_reference_documents() -> dict[str, str]:
    client = get_supabase_client()
    if client is None:
        return {}
    try:
        import PyPDF2
        docs = {}
        for f in client.storage.from_(REFERENCE_BUCKET).list():
            if not f["name"].lower().endswith(".pdf"):
                continue
            try:
                data = client.storage.from_(REFERENCE_BUCKET).download(f["name"])
                extracted = "\n".join(page.extract_text() or "" for page in PyPDF2.PdfReader(io.BytesIO(data)).pages)
                if extracted.strip():
                    docs[f["name"]] = extracted.strip()
            except Exception:
                st.warning(f"⚠️ No se pudo leer '{f['name']}'")
        return docs
    except Exception:
        st.warning("⚠️ Error al cargar documentos de referencia.")
        return {}


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────
def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    return round(weight_kg / (height_cm / 100) ** 2, 2)


def _sanitise(value: str, max_length: int = 500) -> str:
    return "".join(ch for ch in value if ch.isprintable() or ch in ("\n", "\t"))[:max_length]


def _lab_or_na(lab_values, field: str) -> str:
    val = getattr(lab_values, field, None) if lab_values else None
    return f"{val}" if val else "N/A"


def _positive_or_none(val: float):
    return float(val) if val and val > 0 else None


def _parse_csv(raw: str) -> list[str]:
    return [c.strip() for c in raw.split(",") if c.strip()] if raw else []


def _show_error(context: str, error: Exception):
    st.error(f"Error {context}: {error}")


def _gs(key: str):
    """Get session state value with default."""
    return st.session_state.get(key, _STATE_DEFAULTS.get(key))


def _load_patient_and_labs(session, patient_id):
    """Fetch patient + most recent lab values in one go."""
    patient = session.query(Patient).filter_by(id=patient_id).first()
    labs = session.query(LabValue).filter_by(patient_id=patient_id).order_by(LabValue.created_at.desc()).first() if patient else None
    return patient, labs


def extract_file_content(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    try:
        if ext in ("txt", "md"):
            return uploaded_file.read().decode("utf-8")
        if ext == "docx":
            import docx
            return "\n".join(p.text for p in docx.Document(uploaded_file).paragraphs)
        if ext == "pdf":
            import PyPDF2
            return "\n".join(page.extract_text() or "" for page in PyPDF2.PdfReader(uploaded_file).pages)
    except Exception as e:
        _show_error("al procesar archivo", e)
    return ""


# ──────────────────────────────────────────────
# Example plan matching
# ──────────────────────────────────────────────
def find_relevant_examples(patient, lab_values, special_considerations, top_k=2):
    with get_db() as session:
        all_examples = session.query(ExamplePlan).all()
        session.expunge_all()

    if not all_examples:
        return []

    keywords = set()
    if patient.health_conditions:
        keywords.update(c.lower().strip() for c in patient.health_conditions)
    if special_considerations:
        keywords.update(w.strip() for w in special_considerations.lower().split() if len(w.strip()) > 3)
    for threshold, words in AGE_KEYWORDS:
        if patient.age < threshold:
            keywords.update(words)
            break
    for threshold, word in BMI_KEYWORDS:
        if (patient.bmi or 0) < threshold:
            keywords.add(word)
            break

    scored = []
    for ex in all_examples:
        score = sum(3 for t in (ex.tags or []) if t.lower() in keywords)
        if ex.patient_profile:
            score += sum(2 for kw in keywords if kw in ex.patient_profile.lower())
        if ex.plan_content:
            score += sum(1 for kw in keywords if kw in ex.plan_content.lower())
        if score > 0:
            scored.append((score, ex))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [ex for _, ex in scored[:top_k]]


# ──────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────
def _build_diet_prompt(patient, lab_values, special_considerations, relevant_examples):
    examples_text = ""
    if relevant_examples:
        examples_text = "\n\nEJEMPLOS DE FORMATO (SOLO para referencia de estructura y formato, NO copies el contenido):\n\n"
        for idx, ex in enumerate(relevant_examples, 1):
            examples_text += f"--- EJEMPLO DE FORMATO {idx} ---\nPerfil del paciente del ejemplo: {_sanitise(ex.patient_profile or '')}\nFormato de referencia:\n{ex.plan_content}\n\n"

    reference_text = ""
    ref_docs = load_reference_documents()
    if ref_docs:
        reference_text = "\n\nDOCUMENTOS DE REFERENCIA NUTRICIONAL (usa estas tablas y guías para fundamentar las porciones y raciones):\n\n"
        for filename, content in ref_docs.items():
            truncated = content[:3000] + ("..." if len(content) > 3000 else "")
            reference_text += f"--- {filename} ---\n{truncated}\n\n"

    conditions = ", ".join(_sanitise(c) for c in patient.health_conditions) if patient.health_conditions else "Ninguna"
    safe_considerations = _sanitise(special_considerations) if special_considerations else "Ninguna"

    return f"""Eres un nutriólogo experto mexicano. Crea un plan de alimentación integral y personalizado para el siguiente paciente:

Información del Paciente:
- Nombre: {_sanitise(patient.name, 100)}
- Edad: {patient.age} años
- Género: {patient.gender}
- Peso: {patient.weight} kg
- Altura: {patient.height} cm
- IMC: {patient.bmi}
- Condiciones de Salud: {conditions}

Resultados de Laboratorio:
- Glucosa: {_lab_or_na(lab_values, 'glucose')} mg/dL
- Colesterol: {_lab_or_na(lab_values, 'cholesterol')} mg/dL
- Triglicéridos: {_lab_or_na(lab_values, 'triglycerides')} mg/dL
- Hemoglobina: {_lab_or_na(lab_values, 'hemoglobin')} g/dL

Consideraciones Especiales: {safe_considerations}
{reference_text}
{"IMPORTANTE: Basa las porciones y raciones en los documentos de referencia nutricional proporcionados." if ref_docs else ""}
{examples_text}
{"INSTRUCCIÓN CRÍTICA SOBRE LOS EJEMPLOS: Los ejemplos de formato son ÚNICAMENTE para que observes la estructura visual, el tipo de encabezados y la organización general. NO copies, parafrasees ni reutilices los alimentos, cantidades, menús ni texto de los ejemplos. El plan que generes debe ser 100%% original y personalizado para ESTE paciente basándote en sus datos clínicos, condiciones de salud y valores de laboratorio. Si el ejemplo dice 'pollo a la plancha', NO pongas en automatico 'pollo a la plancha' — elige alimentos apropiados para este paciente." if relevant_examples else ""}

Por favor crea un plan detallado que incluya:
1. Desayuno con opciones variadas y multiples ejemplos
2. Colacion con opciones variadas y multiples ejemplos
3. Comida con opciones variadas y multiples ejemplos
4. Cena con opciones variadas y multiples ejemplos
5. Siempre incluye lo siguiente:
"SAL: Modere el consumo de sal, alimentos salados o envasados. Puede utilizar hierbas, especias, ajo, cebolla ó limón para sazonar."
6. Una sección llamada "ELIMINE DE SU DIETA LOS SIGUIENTES ALIMENTOS:" con recomendación de alimentos a no incluir, por ejemplo:
"Azúcar
Frijoles, lentejas, habas, garbanzos, soya,
Jugos naturales
Yogurt saborizados
Pancita, hígado, mollejas, y cualquier tipo de vísceras. Chicharrón, tocino, chorizo, salchicha.
Refrescos y jugos industrializados
Pastelillos"
Las recomendaciones deben estar alineadas a los padecimientos del paciente y recomendaciones nutricionales.
Utiliza como referencia las guías KDOQI de la Academia Mexicana, las guías KDIGO.
Formatea el plan de manera clara y fácil de seguir. Usa alimentos comunes en México.

Al final del plan, incluye una sección llamada "REFERENCIAS Y FUENTES:" donde listes las guías, tablas o documentos en los que te basaste para las recomendaciones (por ejemplo: guías KDOQI, KDIGO, documentos de referencia proporcionados, etc.).

INSTRUCCIÓN DE FORMATO: Este plan se entregará como documento al paciente. NO incluyas frases conversacionales como "¿te gustaría que ajuste algo?", "si necesitas más información", "no dudes en preguntar", "espero que te sea útil" o cualquier otra frase que sugiera una conversación. Termina directamente con el contenido del plan y las referencias."""


# ──────────────────────────────────────────────
# AI generation & validation
# ──────────────────────────────────────────────
def validate_api_key(api_key: str, provider: str) -> bool:
    try:
        if provider == "OpenAI":
            OpenAI(api_key=api_key).chat.completions.create(model="gpt-5.2", messages=[{"role": "user", "content": "hi"}], max_completion_tokens=10)
        else:
            anthropic.Anthropic(api_key=api_key).messages.create(model="claude-sonnet-4-5-20250929", max_tokens=10, messages=[{"role": "user", "content": "hi"}])
        return True
    except Exception:
        return False


def generate_diet_plan(patient, lab_values, special_considerations, api_key, provider):
    prompt = _build_diet_prompt(patient, lab_values, special_considerations, find_relevant_examples(patient, lab_values, special_considerations))
    if provider == "OpenAI":
        resp = OpenAI(api_key=api_key).chat.completions.create(model="gpt-5.2", messages=[{"role": "user", "content": prompt}], max_completion_tokens=4000)
        return resp.choices[0].message.content
    else:
        msg = anthropic.Anthropic(api_key=api_key).messages.create(model="claude-sonnet-4-5-20250929", max_tokens=2500, messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text


# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
def init_session_state():
    for k, v in _STATE_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_form():
    for k, v in _STATE_DEFAULTS.items():
        st.session_state[k] = v


def load_patient_into_state(patient, lab_values):
    st.session_state.update(
        patient_created=True, current_patient_id=patient.id, load_existing_patient=True,
        patient_name=patient.name, patient_age=patient.age,
        patient_gender=GENDER_FROM_DB.get(patient.gender, "Masculino"),
        patient_weight=patient.weight, patient_height=patient.height,
        patient_health_conditions=", ".join(patient.health_conditions) if patient.health_conditions else "",
        patient_glucose=(lab_values.glucose or 0.0) if lab_values else 0.0,
        patient_cholesterol=(lab_values.cholesterol or 0.0) if lab_values else 0.0,
        patient_triglycerides=(lab_values.triglycerides or 0.0) if lab_values else 0.0,
        patient_hemoglobin=(lab_values.hemoglobin or 0.0) if lab_values else 0.0,
    )


# ──────────────────────────────────────────────
# UI: plan card
# ──────────────────────────────────────────────
def render_plan_card(plan, prefix="plan"):
    st.markdown(f"**ID:** {plan.id}  |  **Estado:** {plan.status}  |  **Creado:** {plan.created_at.strftime('%d/%m/%Y %H:%M')}  |  **Actualizado:** {plan.updated_at.strftime('%d/%m/%Y %H:%M')}")
    if plan.special_considerations:
        st.markdown(f"**Consideraciones:** {plan.special_considerations}")
    st.text_area("Plan", value=plan.plan_details, height=300, disabled=True, key=f"{prefix}_c_{plan.id}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("📥 Descargar", data=plan.plan_details, file_name=f"plan_{plan.id}.txt", mime="text/plain", key=f"{prefix}_dl_{plan.id}")
    with c2:
        if st.button("🔄 Editar", key=f"{prefix}_ld_{plan.id}"):
            st.session_state.update(current_plan=plan.plan_details, current_plan_id=plan.id, plan_generated=True)
            st.rerun()
    with c3:
        if st.button("🗑️ Eliminar", key=f"{prefix}_rm_{plan.id}", type="secondary"):
            try:
                with get_db() as s:
                    obj = s.query(DietPlan).filter_by(id=plan.id).first()
                    if obj:
                        s.delete(obj)
                st.rerun()
            except Exception as e:
                _show_error("al eliminar plan", e)


# ══════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════
init_session_state()
st.title("🥗 Asistente de Nutrición con IA - MVP")
st.markdown("---")

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Configuración")
    ai_provider = st.selectbox("Proveedor de IA", ["OpenAI", "Anthropic"])
    secret_key_name = "OPENAI_API_KEY" if ai_provider == "OpenAI" else "ANTHROPIC_API_KEY"
    default_key = st.secrets.get(secret_key_name, "")
    api_key_input = st.text_input(
        f"API Key de {ai_provider}" + (" (en secrets)" if default_key else ""),
        type="password", value="" if default_key else "",
        placeholder="Dejar vacío si está en secrets" if default_key else f"Ingresa tu API key",
    )
    api_key = api_key_input.strip() or default_key

    if api_key:
        cache_key = f"api_valid_{ai_provider}_{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        if cache_key not in st.session_state:
            with st.spinner("Validando..."):
                st.session_state[cache_key] = validate_api_key(api_key, ai_provider)
        if st.session_state[cache_key]:
            st.success(f"✅ API key válida")
        else:
            st.error(f"❌ API key inválida")
            api_key = ""
    else:
        st.warning("⚠️ Sin API key")

    st.markdown("---")
    if st.button("🚪 Cerrar Sesión"):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.subheader("📚 Planes de Ejemplo")
    if st.button("➕ Agregar Plan de Ejemplo"):
        st.session_state.show_add_example = True
    with get_db() as s:
        st.caption(f"{s.query(ExamplePlan).count()} plan(es) de ejemplo")

    st.markdown("---")
    st.subheader("📄 Docs de Referencia")
    ref_docs = load_reference_documents()
    if ref_docs:
        st.success(f"✅ {len(ref_docs)} doc(s)")
        for fname in ref_docs:
            st.caption(f"  • {fname}")
    else:
        st.caption("Agrega PDFs al bucket 'reference-docs' en Supabase.")

# ── Add Example Plan form ──
if st.session_state.get("show_add_example"):
    st.markdown("---")
    st.header("➕ Agregar Plan de Ejemplo")
    with st.form("add_example_form"):
        ex_title = st.text_input("Título *", placeholder="ej: Plan para Diabético Tipo 2")
        ex_profile = st.text_area("Perfil del Paciente *", placeholder="ej: Hombre de 55 años, diabético tipo 2", height=100)
        ex_tags = st.text_input("Etiquetas (comas) *", placeholder="ej: diabetes, sobrepeso")
        st.markdown("**Subir Archivo**")
        uploaded_file = st.file_uploader("Archivo del plan", type=["txt", "md", "docx", "pdf"])
        st.markdown("**O pegar texto**")
        ex_content = st.text_area("Contenido del Plan", height=400)
        c1, c2 = st.columns(2)
        with c1:
            submitted = st.form_submit_button("💾 Guardar", type="primary")
        with c2:
            cancelled = st.form_submit_button("❌ Cancelar")

        if submitted:
            content = extract_file_content(uploaded_file) or ex_content or ""
            if not all([ex_title, ex_profile, ex_tags]):
                st.error("Completa título, perfil y etiquetas")
            elif not content.strip():
                st.error("Sube un archivo o pega el contenido")
            else:
                try:
                    with get_db() as s:
                        s.add(ExamplePlan(title=ex_title, patient_profile=ex_profile, plan_content=content, tags=_parse_csv(ex_tags)))
                    st.success("✅ Ejemplo guardado!")
                    st.session_state.show_add_example = False
                    st.rerun()
                except Exception as e:
                    _show_error("al guardar ejemplo", e)
        if cancelled:
            st.session_state.show_add_example = False
            st.rerun()

    st.markdown("---")
    st.subheader("📋 Ejemplos Existentes")
    with get_db() as s:
        all_examples = s.query(ExamplePlan).order_by(ExamplePlan.created_at.desc()).all()
        s.expunge_all()
    for ex in all_examples:
        with st.expander(f"📄 {ex.title}"):
            st.markdown(f"**Perfil:** {ex.patient_profile}  |  **Etiquetas:** {', '.join(ex.tags or [])}  |  **Creado:** {ex.created_at.strftime('%d/%m/%Y')}")
            st.text_area("Contenido", value=ex.plan_content, height=200, disabled=True, key=f"ex_{ex.id}")
            if st.button("🗑️ Eliminar", key=f"del_ex_{ex.id}"):
                try:
                    with get_db() as s:
                        obj = s.query(ExamplePlan).filter_by(id=ex.id).first()
                        if obj:
                            s.delete(obj)
                    st.rerun()
                except Exception as e:
                    _show_error("al eliminar ejemplo", e)
    if not all_examples:
        st.info("No hay ejemplos aún.")

# ══════════════════════════════════════════════
# Patient selector
# ══════════════════════════════════════════════
st.header("🔍 Buscar Paciente Existente")
with get_db() as s:
    all_patients = s.query(Patient).order_by(Patient.created_at.desc()).all()
    s.expunge_all()

if all_patients:
    options = {"-- Crear Nuevo Paciente --": None} | {f"{p.name} (ID: {p.id}) - {p.age} años": p.id for p in all_patients}
    default_idx = 0
    if st.session_state.current_patient_id:
        for i, pid in enumerate(options.values()):
            if pid == st.session_state.current_patient_id:
                default_idx = i
                break

    selected_id = options[st.selectbox("Selecciona un paciente o crea uno nuevo", list(options.keys()), index=default_idx, key="patient_selector")]

    if selected_id != st.session_state.current_patient_id:
        if selected_id is not None:
            with get_db() as s:
                patient, labs = _load_patient_and_labs(s, selected_id)
                if patient:
                    load_patient_into_state(patient, labs)
            st.rerun()
        elif st.session_state.load_existing_patient:
            reset_form()
            st.rerun()

    if st.session_state.load_existing_patient and st.session_state.current_patient_id:
        st.success(f"📋 Paciente: {_gs('patient_name')} (ID: {st.session_state.current_patient_id})")
    else:
        st.info("📝 Crear nuevo paciente")
else:
    st.info("No hay pacientes. Crea uno nuevo abajo.")

st.markdown("---")

# ══════════════════════════════════════════════
# Patient information
# ══════════════════════════════════════════════
st.header("1️⃣ Información del Paciente")
c1, c2 = st.columns(2)
with c1:
    name = st.text_input("Nombre *", value=_gs("patient_name"))
    age = st.number_input("Edad *", min_value=1, max_value=120, value=int(_gs("patient_age")))
    gv = _gs("patient_gender")
    gender = st.selectbox("Género *", GENDER_OPTIONS, index=GENDER_OPTIONS.index(gv) if gv in GENDER_OPTIONS else 0)
with c2:
    weight = st.number_input("Peso (kg) *", min_value=1.0, max_value=500.0, value=float(_gs("patient_weight")), step=0.1)
    height = st.number_input("Altura (cm) *", min_value=50.0, max_value=250.0, value=float(_gs("patient_height")), step=0.1)
    if weight and height:
        st.metric("IMC", calculate_bmi(weight, height))

health_conditions = st.text_input("Condiciones de Salud (comas)", value=_gs("patient_health_conditions"), placeholder="ej: diabetes, hipertensión")

st.subheader("Resultados de Laboratorio")
c3, c4 = st.columns(2)
with c3:
    glucose = st.number_input("Glucosa (mg/dL)", min_value=0.0, value=float(_gs("patient_glucose")), step=0.1)
    cholesterol = st.number_input("Colesterol (mg/dL)", min_value=0.0, value=float(_gs("patient_cholesterol")), step=0.1)
with c4:
    triglycerides = st.number_input("Triglicéridos (mg/dL)", min_value=0.0, value=float(_gs("patient_triglycerides")), step=0.1)
    hemoglobin = st.number_input("Hemoglobina (g/dL)", min_value=0.0, value=float(_gs("patient_hemoglobin")), step=0.1)

has_labs = any([glucose, cholesterol, triglycerides, hemoglobin])
lab_kw = dict(test_date=datetime.now().strftime("%Y-%m-%d"), glucose=_positive_or_none(glucose), cholesterol=_positive_or_none(cholesterol), triglycerides=_positive_or_none(triglycerides), hemoglobin=_positive_or_none(hemoglobin))

cb1, cb2 = st.columns(2)
with cb1:
    if not st.session_state.load_existing_patient:
        if st.button("💾 Crear Paciente", type="primary", disabled=st.session_state.patient_created):
            if not (name and age and weight and height):
                st.error("Completa los campos requeridos (*)")
            else:
                try:
                    with get_db() as s:
                        p = Patient(name=name, age=int(age), gender=GENDER_TO_DB.get(gender, "other"), weight=float(weight), height=float(height), health_conditions=_parse_csv(health_conditions), bmi=calculate_bmi(weight, height))
                        s.add(p)
                        s.flush()
                        if has_labs:
                            s.add(LabValue(patient_id=p.id, **lab_kw))
                        st.session_state.update(patient_created=True, current_patient_id=p.id)
                    st.success(f"✅ Paciente creado (ID: {st.session_state.current_patient_id})")
                except Exception as e:
                    _show_error("al crear paciente", e)

with cb2:
    if st.session_state.load_existing_patient:
        if st.button("🔄 Actualizar Paciente", type="primary"):
            if not (name and age and weight and height):
                st.error("Completa los campos requeridos (*)")
            else:
                try:
                    with get_db() as s:
                        p = s.query(Patient).filter_by(id=st.session_state.current_patient_id).first()
                        if not p:
                            st.error("Paciente no encontrado")
                        else:
                            p.name, p.age, p.gender = name, int(age), GENDER_TO_DB.get(gender, "other")
                            p.weight, p.height = float(weight), float(height)
                            p.health_conditions, p.bmi, p.updated_at = _parse_csv(health_conditions), calculate_bmi(weight, height), _utcnow()
                            if has_labs:
                                s.add(LabValue(patient_id=p.id, **lab_kw))
                            st.session_state.update(patient_name=name, patient_age=age, patient_gender=gender, patient_weight=weight, patient_height=height, patient_health_conditions=health_conditions, patient_glucose=glucose, patient_cholesterol=cholesterol, patient_triglycerides=triglycerides, patient_hemoglobin=hemoglobin)
                            st.success(f"✅ Actualizado (ID: {p.id})")
                except Exception as e:
                    _show_error("al actualizar paciente", e)

st.markdown("---")

# ══════════════════════════════════════════════
# Lab history
# ══════════════════════════════════════════════
if st.session_state.patient_created and st.session_state.current_patient_id:
    st.header("🔬 Historial de Laboratorio")
    with get_db() as s:
        all_labs = s.query(LabValue).filter_by(patient_id=st.session_state.current_patient_id).order_by(LabValue.test_date.asc()).all()
        s.expunge_all()

    if all_labs:
        df = pd.DataFrame([{"Fecha": l.test_date, "Glucosa (mg/dL)": l.glucose, "Colesterol (mg/dL)": l.cholesterol, "Triglicéridos (mg/dL)": l.triglycerides, "Hemoglobina (g/dL)": l.hemoglobin} for l in all_labs])
        with st.expander("📊 Tabla de resultados", expanded=False):
            st.dataframe(df, use_container_width=True, hide_index=True)

        if len(all_labs) >= 2:
            st.subheader("📈 Tendencias")
            cc1, cc2 = st.columns(2)
            for idx, (metric, info) in enumerate(LAB_METRICS.items()):
                with (cc1 if idx % 2 == 0 else cc2):
                    mdf = df[["Fecha", metric]].dropna(subset=[metric])
                    if len(mdf) >= 2:
                        st.markdown(f"**{metric}**")
                        st.line_chart(mdf.set_index("Fecha"), color=info["color"])
                        latest, prev = mdf[metric].iloc[-1], mdf[metric].iloc[-2]
                        lo, hi = info["normal"]
                        st.metric(f"Último ({'✅' if lo <= latest <= hi else '⚠️'})", f"{latest:.1f}", f"{latest - prev:+.1f}", delta_color="inverse" if metric != "Hemoglobina (g/dL)" else "normal")
                        st.caption(f"Normal: {lo}–{hi}")
        else:
            st.info("2+ registros necesarios para tendencias.")
    else:
        st.info("Sin registros de laboratorio aún.")

st.markdown("---")

# ══════════════════════════════════════════════
# Generate plan
# ══════════════════════════════════════════════
st.header("2️⃣ Consideraciones Especiales y Generar Plan")
special_considerations = st.text_area("Consideraciones Especiales", placeholder="Alergias, preferencias, restricciones...", height=100)

if st.button("🤖 Generar Plan", type="primary", disabled=not st.session_state.patient_created):
    if not api_key:
        st.error(f"⚠️ Se requiere API key de {ai_provider}.")
    else:
        try:
            with st.spinner(f"Generando plan ({ai_provider})..."):
                with get_db() as s:
                    patient, labs = _load_patient_and_labs(s, st.session_state.current_patient_id)
                    plan_text = generate_diet_plan(patient, labs, special_considerations, api_key, ai_provider)
                    plan = DietPlan(patient_id=patient.id, plan_details=plan_text, special_considerations=special_considerations, status="active")
                    s.add(plan)
                    s.flush()
                    st.session_state.update(plan_generated=True, current_plan=plan_text, current_plan_id=plan.id)
                st.success(f"✅ Plan generado (ID: {st.session_state.current_plan_id})")
                st.rerun()
        except Exception as e:
            _show_error("al generar plan", e)

st.markdown("---")

# ══════════════════════════════════════════════
# Past plans
# ══════════════════════════════════════════════
if st.session_state.patient_created and st.session_state.current_patient_id:
    st.header("📚 Planes Anteriores")
    with get_db() as s:
        past_plans = s.query(DietPlan).filter_by(patient_id=st.session_state.current_patient_id).order_by(DietPlan.created_at.desc()).all()
        s.expunge_all()
    if past_plans:
        st.info(f"{len(past_plans)} plan(es)")
        for idx, plan in enumerate(past_plans):
            with st.expander(f"📋 Plan #{idx + 1} - {plan.created_at.strftime('%d/%m/%Y %H:%M')}", expanded=(len(past_plans) == 1)):
                render_plan_card(plan, prefix=f"p{idx}")
    else:
        st.info("Sin planes guardados.")

st.markdown("---")

# ══════════════════════════════════════════════
# Current plan & modifications
# ══════════════════════════════════════════════
if st.session_state.plan_generated and st.session_state.current_plan:
    st.header("3️⃣ Plan Generado")
    st.text_area("Plan", value=st.session_state.current_plan, height=400, disabled=True)
    st.download_button("📥 Descargar", data=st.session_state.current_plan, file_name=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", mime="text/plain")

    st.markdown("---")
    st.subheader("Modificar y Regenerar")
    modifications = st.text_area("Modificaciones", placeholder="ej: Más proteína, menos carbohidratos", height=100)

    if st.button("🔄 Regenerar Plan", type="secondary", disabled=not modifications):
        if not api_key:
            st.error(f"⚠️ Se requiere API key de {ai_provider}.")
        else:
            try:
                with st.spinner(f"Regenerando ({ai_provider})..."):
                    with get_db() as s:
                        patient, labs = _load_patient_and_labs(s, st.session_state.current_patient_id)
                        mod = f"{special_considerations}\n\nModificaciones: {modifications}"
                        new_text = generate_diet_plan(patient, labs, mod, api_key, ai_provider)
                        existing = s.query(DietPlan).filter_by(id=st.session_state.current_plan_id).first()
                        if existing:
                            existing.plan_details, existing.special_considerations, existing.updated_at = new_text, mod, _utcnow()
                            st.session_state.current_plan = new_text
                            st.success("✅ Regenerado!")
                        else:
                            st.error("Plan no encontrado")
                    st.rerun()
            except Exception as e:
                _show_error("al regenerar plan", e)

# ── Footer ──
st.markdown("---")
st.caption("💡 Usa el selector en la parte superior para cambiar de paciente.")
