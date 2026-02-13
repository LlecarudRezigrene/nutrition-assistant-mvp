import streamlit as st
import os
import hashlib
import hmac
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime, Text, ForeignKey, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship
from datetime import datetime, timezone
import pandas as pd
from openai import OpenAI
import anthropic

# ──────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Asistente de Nutrición con IA",
    page_icon="🥗",
    layout="wide",
)

# ──────────────────────────────────────────────
# Authentication
# ──────────────────────────────────────────────
# Add credentials in .streamlit/secrets.toml:
#
# [auth]
# username = "your_username"
# password_hash = "paste_the_sha256_hash_here"
#
# Generate a hash with:
#   python -c "import hashlib; print(hashlib.sha256('YOUR_PASSWORD'.encode()).hexdigest())"


def _check_credentials(username: str, password: str) -> bool:
    """Compare credentials against secrets using constant-time comparison."""
    try:
        stored_user = st.secrets["auth"]["username"]
        stored_hash = st.secrets["auth"]["password_hash"]
    except KeyError:
        st.error("❌ Credenciales no configuradas en secrets. Agrega [auth] con username y password_hash.")
        return False

    password_hash = hashlib.sha256(password.encode()).hexdigest()
    user_ok = hmac.compare_digest(username, stored_user)
    pass_ok = hmac.compare_digest(password_hash, stored_hash)
    return user_ok and pass_ok


def login_page():
    """Render login form and return True if authenticated."""
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

AGE_KEYWORDS = [
    (18, ["adolescente", "joven"]),
    (30, ["adulto joven"]),
    (60, ["adulto"]),
    (999, ["adulto mayor", "tercera edad"]),
]

BMI_KEYWORDS = [
    (18.5, "bajo peso"),
    (25, "peso normal"),
    (30, "sobrepeso"),
    (999, "obesidad"),
]


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
# Database initialisation & session helper
# ──────────────────────────────────────────────
@st.cache_resource
def init_db():
    if "DATABASE_URL" not in st.secrets:
        st.error("❌ ERROR: DATABASE_URL no está configurado en los secrets de Streamlit.")
        st.stop()

    database_url = st.secrets["DATABASE_URL"]

    if not database_url.startswith("postgresql://"):
        st.error("❌ ERROR: DATABASE_URL debe comenzar con 'postgresql://'")
        st.stop()

    try:
        engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=3600)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        Base.metadata.create_all(engine)
        return sessionmaker(bind=engine)
    except Exception as e:
        st.error(f"❌ ERROR al conectar con la base de datos: {e}")
        st.stop()


SessionFactory = init_db()


@contextmanager
def get_db():
    """Context manager that guarantees session cleanup on success or failure."""
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────
def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 2)


def _sanitise_for_prompt(value: str, max_length: int = 500) -> str:
    """Strip control characters and limit length to prevent prompt injection."""
    cleaned = "".join(ch for ch in value if ch.isprintable() or ch in ("\n", "\t"))
    return cleaned[:max_length]


def _lab_value_or_na(lab_values, field: str) -> str:
    val = getattr(lab_values, field, None) if lab_values else None
    return f"{val}" if val else "N/A"


def _lab_unit(field: str) -> str:
    return "g/dL" if field == "hemoglobin" else "mg/dL"


def _positive_or_none(val: float):
    """Return value only if > 0, else None (for DB storage)."""
    return float(val) if val and val > 0 else None


# ──────────────────────────────────────────────
# RAG: find relevant example plans
# ──────────────────────────────────────────────
def find_relevant_examples(patient, lab_values, special_considerations, top_k=2):
    """Score example plans by keyword overlap with patient profile."""
    with get_db() as session:
        all_examples = session.query(ExamplePlan).all()
        # Detach from session so we can use them after close
        session.expunge_all()

    if not all_examples:
        return []

    # Build search keywords from patient data
    keywords = set()

    if patient.health_conditions:
        for cond in patient.health_conditions:
            keywords.add(cond.lower().strip())

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

    # Score each example (tags=3pts, profile=2pts, content=1pt)
    scored = []
    for ex in all_examples:
        score = 0
        if ex.tags:
            score += sum(3 for t in ex.tags if t.lower() in keywords)
        if ex.patient_profile:
            profile = ex.patient_profile.lower()
            score += sum(2 for kw in keywords if kw in profile)
        if ex.plan_content:
            content = ex.plan_content.lower()
            score += sum(1 for kw in keywords if kw in content)
        if score > 0:
            scored.append((score, ex))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [ex for _, ex in scored[:top_k]]


# ──────────────────────────────────────────────
# Prompt builder (shared between providers)
# ──────────────────────────────────────────────
def _build_diet_prompt(patient, lab_values, special_considerations, relevant_examples):
    """Build the diet plan prompt — single source of truth."""
    examples_text = ""
    if relevant_examples:
        examples_text = "\n\nEJEMPLOS DE REFERENCIA (usa estos como guía de estilo y formato):\n\n"
        for idx, ex in enumerate(relevant_examples, 1):
            examples_text += (
                f"--- EJEMPLO {idx} ---\n"
                f"Perfil del paciente: {_sanitise_for_prompt(ex.patient_profile or '')}\n"
                f"Plan:\n{ex.plan_content}\n\n"
            )

    conditions = ", ".join(patient.health_conditions) if patient.health_conditions else "Ninguna"
    safe_considerations = _sanitise_for_prompt(special_considerations) if special_considerations else "Ninguna"

    return f"""Eres un nutriólogo experto mexicano. Crea un plan de alimentación integral y personalizado para el siguiente paciente:

Información del Paciente:
- Nombre: {_sanitise_for_prompt(patient.name, 100)}
- Edad: {patient.age} años
- Género: {patient.gender}
- Peso: {patient.weight} kg
- Altura: {patient.height} cm
- IMC: {patient.bmi}
- Condiciones de Salud: {conditions}

Resultados de Laboratorio:
- Glucosa: {_lab_value_or_na(lab_values, 'glucose')} mg/dL
- Colesterol: {_lab_value_or_na(lab_values, 'cholesterol')} mg/dL
- Triglicéridos: {_lab_value_or_na(lab_values, 'triglycerides')} mg/dL
- Hemoglobina: {_lab_value_or_na(lab_values, 'hemoglobin')} g/dL

Consideraciones Especiales: {safe_considerations}
{examples_text}
{"IMPORTANTE: Usa los ejemplos de referencia como guía para el estilo, formato y estructura del plan. Adapta el contenido específicamente para este paciente, pero mantén un estilo similar." if relevant_examples else ""}

Por favor crea un plan detallado que incluya:
1. Desayuno con opciones variadas y multiples ejemplos
2. Colacion con opciones variadas y multiples ejemplos
3. Comida con opciones variadas y multiples ejemplos
4. Cena con opciones variadas y multiples ejemplos
5. Siempre incluye lo siguiente:
"SAL: Modere el consumo de sal, alimentos salados o envasados. Puede utilizar 
hierbas, especias, ajo, cebolla ó limón para sazonar." 
6. Una secciòn llamada "ELIMINE   DE   SU  DIETA  LOS   SIGUIENTES   ALIMENTOS: " con recomendacion de alimentos a no incluir, por ejemplo:
 - - - - - - 
Azúcar 
Frijoles, lentejas, habas, garbanzos, soya,  
Jugos naturales 
Yogurt saborizados 
Pancita, hígado, mollejas, y cualquier tipo de vísceras. Chicharrón, 
tocino, chorizo, salchicha. 
Refrescos y jugos industrializados 
Pastelillos

Formatea el plan de manera clara y fácil de seguir. Usa alimentos comunes en México."""


# ──────────────────────────────────────────────
# AI generation
# ──────────────────────────────────────────────
def generate_diet_plan(patient, lab_values, special_considerations, api_key, provider):
    """Unified generation function for both providers."""
    relevant_examples = find_relevant_examples(patient, lab_values, special_considerations)
    prompt = _build_diet_prompt(patient, lab_values, special_considerations, relevant_examples)

    if provider == "OpenAI":
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2500,
        )
        return response.choices[0].message.content
    else:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


# ──────────────────────────────────────────────
# Session state helpers
# ──────────────────────────────────────────────
_DEFAULT_STATE = {
    "patient_created": False,
    "plan_generated": False,
    "current_patient_id": None,
    "current_plan": None,
    "current_plan_id": None,
    "load_existing_patient": False,
    "show_add_example": False,
}

_PATIENT_FIELDS = {
    "patient_name": "",
    "patient_age": 30,
    "patient_gender": "Masculino",
    "patient_weight": 70.0,
    "patient_height": 170.0,
    "patient_health_conditions": "",
    "patient_glucose": 0.0,
    "patient_cholesterol": 0.0,
    "patient_triglycerides": 0.0,
    "patient_hemoglobin": 0.0,
}


def init_session_state():
    for key, default in {**_DEFAULT_STATE, **_PATIENT_FIELDS}.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_form():
    for key, default in {**_DEFAULT_STATE, **_PATIENT_FIELDS}.items():
        st.session_state[key] = default


def load_patient_into_state(patient, lab_values):
    """Populate session_state from a Patient + LabValue record."""
    st.session_state.patient_created = True
    st.session_state.current_patient_id = patient.id
    st.session_state.load_existing_patient = True
    st.session_state.patient_name = patient.name
    st.session_state.patient_age = patient.age
    st.session_state.patient_gender = GENDER_FROM_DB.get(patient.gender, "Masculino")
    st.session_state.patient_weight = patient.weight
    st.session_state.patient_height = patient.height
    st.session_state.patient_health_conditions = (
        ", ".join(patient.health_conditions) if patient.health_conditions else ""
    )
    st.session_state.patient_glucose = (lab_values.glucose or 0.0) if lab_values else 0.0
    st.session_state.patient_cholesterol = (lab_values.cholesterol or 0.0) if lab_values else 0.0
    st.session_state.patient_triglycerides = (lab_values.triglycerides or 0.0) if lab_values else 0.0
    st.session_state.patient_hemoglobin = (lab_values.hemoglobin or 0.0) if lab_values else 0.0


# ──────────────────────────────────────────────
# UI component: display a single diet plan
# ──────────────────────────────────────────────
def render_plan_card(plan, prefix="plan"):
    """Reusable component to display a diet plan with actions."""
    st.markdown(f"**ID del Plan:** {plan.id}")
    st.markdown(f"**Estado:** {plan.status}")
    st.markdown(f"**Fecha de creación:** {plan.created_at.strftime('%d de %B de %Y a las %H:%M')}")
    st.markdown(f"**Última actualización:** {plan.updated_at.strftime('%d de %B de %Y a las %H:%M')}")

    if plan.special_considerations:
        st.markdown("**Consideraciones Especiales:**")
        st.text(plan.special_considerations)

    st.markdown("---")
    st.markdown("**Contenido del Plan:**")
    st.text_area(
        "Plan",
        value=plan.plan_details,
        height=300,
        disabled=True,
        key=f"{prefix}_content_{plan.id}",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="📥 Descargar",
            data=plan.plan_details,
            file_name=f"plan_{plan.id}_{plan.created_at.strftime('%Y%m%d')}.txt",
            mime="text/plain",
            key=f"{prefix}_download_{plan.id}",
        )
    with col2:
        if st.button("🔄 Cargar para Editar", key=f"{prefix}_load_{plan.id}"):
            st.session_state.current_plan = plan.plan_details
            st.session_state.current_plan_id = plan.id
            st.session_state.plan_generated = True
            st.success("Plan cargado para editar")
            st.rerun()
    with col3:
        if st.button("🗑️ Eliminar Plan", key=f"{prefix}_delete_{plan.id}", type="secondary"):
            try:
                with get_db() as session:
                    plan_to_delete = session.query(DietPlan).filter_by(id=plan.id).first()
                    if plan_to_delete:
                        session.delete(plan_to_delete)
                        st.success("Plan eliminado exitosamente")
                    else:
                        st.error("Plan no encontrado")
                st.rerun()
            except Exception as e:
                st.error(f"Error al eliminar plan: {e}")


# ──────────────────────────────────────────────
# UI component: file upload processor
# ──────────────────────────────────────────────
def extract_file_content(uploaded_file) -> str:
    """Extract text from uploaded .txt, .md, .docx, or .pdf."""
    if uploaded_file is None:
        return ""

    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()

    try:
        if ext in ("txt", "md"):
            return uploaded_file.read().decode("utf-8")
        elif ext == "docx":
            import docx
            doc = docx.Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs)
        elif ext == "pdf":
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        st.error(f"Error al procesar archivo: {e}")
    return ""


# ══════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════
init_session_state()

st.title("🥗 Asistente de Nutrición con IA - MVP")
st.markdown("---")

# ── Sidebar: API config ──
with st.sidebar:
    st.header("⚙️ Configuración")

    ai_provider = st.selectbox("Selecciona el Proveedor de IA", ["OpenAI", "Anthropic"])

    api_key = st.text_input(
        f"Ingresa tu API Key de {ai_provider}",
        type="password",
        help=f"Obtén tu API key en {ai_provider.lower()}.com",
    )

    st.markdown("---")
    st.caption("Tu API key solo se usa para esta sesión y nunca se almacena.")

    st.markdown("---")
    if st.button("🚪 Cerrar Sesión"):
        st.session_state.clear()
        st.rerun()

# Gate: require API key
if not api_key:
    st.warning(f"👈 Por favor ingresa tu API key de {ai_provider} en la barra lateral para comenzar.")
    st.info(
        "**Cómo obtener una API key:**\n"
        "- **OpenAI**: Visita [platform.openai.com](https://platform.openai.com) → API Keys\n"
        "- **Anthropic**: Visita [console.anthropic.com](https://console.anthropic.com) → API Keys"
    )
    st.stop()

# ── Sidebar: Example Plans management ──
with st.sidebar:
    st.markdown("---")
    st.subheader("📚 Planes de Ejemplo")

    if st.button("➕ Agregar Plan de Ejemplo"):
        st.session_state.show_add_example = True

    with get_db() as session:
        example_count = session.query(ExamplePlan).count()
    st.caption(f"Tienes {example_count} plan(es) de ejemplo guardado(s)")

# ── Add Example Plan form (shown in main area) ──
if st.session_state.get("show_add_example", False):
    st.markdown("---")
    st.header("➕ Agregar Plan de Ejemplo")

    with st.form("add_example_form"):
        example_title = st.text_input("Título del Plan *", placeholder="ej: Plan para Diabético Tipo 2")
        example_profile = st.text_area(
            "Perfil del Paciente *",
            placeholder="ej: Hombre de 55 años, diabético tipo 2, sedentario, IMC 28",
            height=100,
        )
        example_tags_input = st.text_input(
            "Etiquetas (separadas por comas) *",
            placeholder="ej: diabetes, sobrepeso, sedentario, hipertensión",
        )

        st.markdown("**Opción 1: Subir Archivo**")
        uploaded_file = st.file_uploader(
            "Sube un archivo con el plan de ejemplo",
            type=["txt", "md", "docx", "pdf"],
            help="Formatos soportados: .txt, .md, .docx, .pdf",
        )
        st.markdown("**O**")
        st.markdown("**Opción 2: Pegar Texto Directamente**")
        example_content = st.text_area(
            "Contenido del Plan",
            placeholder="Pega aquí un plan de ejemplo completo...",
            height=400,
        )

        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("💾 Guardar Ejemplo", type="primary")
        with col2:
            cancelled = st.form_submit_button("❌ Cancelar")

        if submitted:
            final_content = extract_file_content(uploaded_file) or example_content or ""

            if not example_title or not example_profile or not example_tags_input:
                st.error("Por favor completa el título, perfil del paciente y etiquetas")
            elif not final_content.strip():
                st.error("Por favor sube un archivo O pega el contenido del plan")
            else:
                try:
                    tags_list = [t.strip() for t in example_tags_input.split(",") if t.strip()]
                    with get_db() as session:
                        session.add(ExamplePlan(
                            title=example_title,
                            patient_profile=example_profile,
                            plan_content=final_content,
                            tags=tags_list,
                        ))
                    st.success("✅ Plan de ejemplo guardado exitosamente!")
                    st.session_state.show_add_example = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al guardar ejemplo: {e}")

        if cancelled:
            st.session_state.show_add_example = False
            st.rerun()

    st.markdown("---")

    # Show existing examples
    st.subheader("📋 Planes de Ejemplo Existentes")
    with get_db() as session:
        all_examples = session.query(ExamplePlan).order_by(ExamplePlan.created_at.desc()).all()
        session.expunge_all()

    if all_examples:
        for example in all_examples:
            with st.expander(f"📄 {example.title}", expanded=False):
                st.markdown(f"**ID:** {example.id}")
                st.markdown(f"**Perfil:** {example.patient_profile}")
                st.markdown(f"**Etiquetas:** {', '.join(example.tags) if example.tags else 'Sin etiquetas'}")
                st.markdown(f"**Creado:** {example.created_at.strftime('%d/%m/%Y %H:%M')}")
                st.text_area(
                    "Contenido",
                    value=example.plan_content,  # ← BUG FIX: was 'example.content'
                    height=200,
                    disabled=True,
                    key=f"example_content_{example.id}",
                )
                if st.button("🗑️ Eliminar", key=f"delete_example_{example.id}"):
                    try:
                        with get_db() as session:
                            obj = session.query(ExamplePlan).filter_by(id=example.id).first()
                            if obj:
                                session.delete(obj)
                        st.success("Ejemplo eliminado")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("No hay planes de ejemplo. Agrega algunos para mejorar la calidad de los planes generados.")

# ══════════════════════════════════════════════
# Section 0: Search existing patient
# ══════════════════════════════════════════════
st.header("🔍 Buscar Paciente Existente")

with get_db() as session:
    all_patients = session.query(Patient).order_by(Patient.created_at.desc()).all()
    session.expunge_all()

if all_patients:
    patient_options = {"-- Crear Nuevo Paciente --": None}
    for p in all_patients:
        patient_options[f"{p.name} (ID: {p.id}) - {p.age} años"] = p.id

    # Determine default selection
    default_index = 0
    if st.session_state.current_patient_id:
        for idx, (_, pid) in enumerate(patient_options.items()):
            if pid == st.session_state.current_patient_id:
                default_index = idx
                break

    selected_patient = st.selectbox(
        "Selecciona un paciente existente o crea uno nuevo",
        options=list(patient_options.keys()),
        index=default_index,
        key="patient_selector",
    )
    selected_patient_id = patient_options[selected_patient]

    # Handle selection change
    if selected_patient_id != st.session_state.current_patient_id:
        if selected_patient_id is not None:
            with get_db() as session:
                patient = session.query(Patient).filter_by(id=selected_patient_id).first()
                lab_vals = (
                    session.query(LabValue)
                    .filter_by(patient_id=patient.id)
                    .order_by(LabValue.created_at.desc())
                    .first()
                )
                if patient:
                    load_patient_into_state(patient, lab_vals)
            st.rerun()
        else:
            if st.session_state.load_existing_patient:
                reset_form()
                st.rerun()

    if st.session_state.load_existing_patient and st.session_state.current_patient_id:
        st.success(
            f"📋 Paciente cargado: {st.session_state.get('patient_name', 'Desconocido')} "
            f"(ID: {st.session_state.current_patient_id})"
        )
    else:
        st.info("📝 Modo: Crear nuevo paciente")
else:
    st.info("No hay pacientes en la base de datos. Crea uno nuevo abajo.")

st.markdown("---")

# ══════════════════════════════════════════════
# Section 1: Patient information
# ══════════════════════════════════════════════
st.header("1️⃣ Información del Paciente")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Nombre *", value=st.session_state.get("patient_name", ""))
    age = st.number_input("Edad *", min_value=1, max_value=120, value=int(st.session_state.get("patient_age", 30)))
    gender_val = st.session_state.get("patient_gender", "Masculino")
    gender_index = GENDER_OPTIONS.index(gender_val) if gender_val in GENDER_OPTIONS else 0
    gender = st.selectbox("Género *", GENDER_OPTIONS, index=gender_index)

with col2:
    weight = st.number_input(
        "Peso (kg) *", min_value=1.0, max_value=500.0,
        value=float(st.session_state.get("patient_weight", 70.0)), step=0.1,
    )
    height = st.number_input(
        "Altura (cm) *", min_value=50.0, max_value=250.0,
        value=float(st.session_state.get("patient_height", 170.0)), step=0.1,
    )
    if weight and height:
        bmi = calculate_bmi(weight, height)
        st.metric("IMC", bmi)

health_conditions = st.text_input(
    "Condiciones de Salud (separadas por comas)",
    value=st.session_state.get("patient_health_conditions", ""),
    placeholder="ej: diabetes, hipertensión, enfermedad celíaca",
)

st.subheader("Resultados de Laboratorio")
col3, col4 = st.columns(2)

with col3:
    glucose = st.number_input("Glucosa (mg/dL)", min_value=0.0, value=float(st.session_state.get("patient_glucose", 0.0)), step=0.1)
    cholesterol = st.number_input("Colesterol (mg/dL)", min_value=0.0, value=float(st.session_state.get("patient_cholesterol", 0.0)), step=0.1)

with col4:
    triglycerides = st.number_input("Triglicéridos (mg/dL)", min_value=0.0, value=float(st.session_state.get("patient_triglycerides", 0.0)), step=0.1)
    hemoglobin = st.number_input("Hemoglobina (g/dL)", min_value=0.0, value=float(st.session_state.get("patient_hemoglobin", 0.0)), step=0.1)

# ── Create / Update buttons ──
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if not st.session_state.load_existing_patient:
        if st.button("💾 Crear Paciente y Guardar Datos", type="primary", disabled=st.session_state.patient_created):
            if not name or not age or not weight or not height:
                st.error("Por favor completa todos los campos requeridos (marcados con *)")
            else:
                try:
                    conditions_list = [c.strip() for c in health_conditions.split(",") if c.strip()] if health_conditions else []
                    bmi_value = calculate_bmi(weight, height)
                    gender_db = GENDER_TO_DB.get(gender, "other")

                    with get_db() as session:
                        new_patient = Patient(
                            name=name, age=int(age), gender=gender_db,
                            weight=float(weight), height=float(height),
                            health_conditions=conditions_list, bmi=bmi_value,
                        )
                        session.add(new_patient)
                        session.flush()  # get the id

                        if any([glucose, cholesterol, triglycerides, hemoglobin]):
                            session.add(LabValue(
                                patient_id=new_patient.id,
                                test_date=datetime.now().strftime("%Y-%m-%d"),
                                glucose=_positive_or_none(glucose),
                                cholesterol=_positive_or_none(cholesterol),
                                triglycerides=_positive_or_none(triglycerides),
                                hemoglobin=_positive_or_none(hemoglobin),
                            ))

                        st.session_state.patient_created = True
                        st.session_state.current_patient_id = new_patient.id

                    st.success(f"✅ ¡Paciente creado exitosamente! (ID: {st.session_state.current_patient_id})")
                except Exception as e:
                    st.error(f"Error al crear paciente: {e}")

with col_btn2:
    if st.session_state.load_existing_patient:
        if st.button("🔄 Actualizar Datos del Paciente", type="primary"):
            if not name or not age or not weight or not height:
                st.error("Por favor completa todos los campos requeridos (marcados con *)")
            else:
                try:
                    with get_db() as session:
                        patient = session.query(Patient).filter_by(id=st.session_state.current_patient_id).first()
                        if not patient:
                            st.error("No se encontró el paciente")
                        else:
                            conditions_list = [c.strip() for c in health_conditions.split(",") if c.strip()] if health_conditions else []
                            patient.name = name
                            patient.age = int(age)
                            patient.gender = GENDER_TO_DB.get(gender, "other")
                            patient.weight = float(weight)
                            patient.height = float(height)
                            patient.health_conditions = conditions_list
                            patient.bmi = calculate_bmi(weight, height)
                            patient.updated_at = _utcnow()

                            # Always create a NEW lab record to preserve history
                            if any([glucose, cholesterol, triglycerides, hemoglobin]):
                                session.add(LabValue(
                                    patient_id=patient.id,
                                    test_date=datetime.now().strftime("%Y-%m-%d"),
                                    glucose=_positive_or_none(glucose),
                                    cholesterol=_positive_or_none(cholesterol),
                                    triglycerides=_positive_or_none(triglycerides),
                                    hemoglobin=_positive_or_none(hemoglobin),
                                ))

                            # Sync session state
                            st.session_state.patient_name = name
                            st.session_state.patient_age = age
                            st.session_state.patient_gender = gender
                            st.session_state.patient_weight = weight
                            st.session_state.patient_height = height
                            st.session_state.patient_health_conditions = health_conditions
                            st.session_state.patient_glucose = glucose
                            st.session_state.patient_cholesterol = cholesterol
                            st.session_state.patient_triglycerides = triglycerides
                            st.session_state.patient_hemoglobin = hemoglobin

                            st.success(f"✅ ¡Datos actualizados! (ID: {patient.id})")
                except Exception as e:
                    st.error(f"Error al actualizar paciente: {e}")

st.markdown("---")

# ══════════════════════════════════════════════
# Section: Lab value history
# ══════════════════════════════════════════════
if st.session_state.patient_created and st.session_state.current_patient_id:
    st.header("🔬 Historial de Laboratorio")

    with get_db() as session:
        all_labs = (
            session.query(LabValue)
            .filter_by(patient_id=st.session_state.current_patient_id)
            .order_by(LabValue.test_date.asc())
            .all()
        )
        session.expunge_all()

    if len(all_labs) >= 1:
        # Build dataframe from lab records
        lab_records = []
        for lab in all_labs:
            lab_records.append({
                "Fecha": lab.test_date,
                "Glucosa (mg/dL)": lab.glucose,
                "Colesterol (mg/dL)": lab.cholesterol,
                "Triglicéridos (mg/dL)": lab.triglycerides,
                "Hemoglobina (g/dL)": lab.hemoglobin,
            })

        df = pd.DataFrame(lab_records)

        # Show data table
        with st.expander("📊 Ver tabla de resultados", expanded=False):
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Show trend charts if there are at least 2 records
        if len(all_labs) >= 2:
            st.subheader("📈 Tendencias")

            lab_metrics = {
                "Glucosa (mg/dL)": {"normal": (70, 100), "color": "#FF6B6B"},
                "Colesterol (mg/dL)": {"normal": (0, 200), "color": "#4ECDC4"},
                "Triglicéridos (mg/dL)": {"normal": (0, 150), "color": "#45B7D1"},
                "Hemoglobina (g/dL)": {"normal": (12, 17), "color": "#96CEB4"},
            }

            chart_col1, chart_col2 = st.columns(2)

            for idx, (metric, info) in enumerate(lab_metrics.items()):
                col = chart_col1 if idx % 2 == 0 else chart_col2
                with col:
                    metric_df = df[["Fecha", metric]].dropna(subset=[metric])
                    if len(metric_df) >= 2:
                        st.markdown(f"**{metric}**")
                        st.line_chart(metric_df.set_index("Fecha"), color=info["color"])

                        # Show latest vs previous comparison
                        latest = metric_df[metric].iloc[-1]
                        previous = metric_df[metric].iloc[-2]
                        delta = latest - previous
                        lo, hi = info["normal"]
                        status = "✅ Normal" if lo <= latest <= hi else "⚠️ Fuera de rango"
                        st.metric(
                            label=f"Último valor ({status})",
                            value=f"{latest:.1f}",
                            delta=f"{delta:+.1f} vs anterior",
                            delta_color="inverse" if metric != "Hemoglobina (g/dL)" else "normal",
                        )
                        st.caption(f"Rango normal: {lo}–{hi}")
        else:
            st.info("Se necesitan al menos 2 registros de laboratorio para mostrar tendencias. Los valores se irán acumulando en cada visita.")
    else:
        st.info("No hay registros de laboratorio para este paciente aún.")

st.markdown("---")

# ══════════════════════════════════════════════
# Section 2: Special considerations & generate plan
# ══════════════════════════════════════════════
st.header("2️⃣ Consideraciones Especiales y Generar Plan")

special_considerations = st.text_area(
    "Consideraciones Especiales",
    placeholder="Alergias, preferencias alimentarias, restricciones dietéticas, consideraciones culturales...",
    height=100,
)

if st.button("🤖 Generar Plan de Alimentación", type="primary", disabled=not st.session_state.patient_created):
    if not st.session_state.patient_created:
        st.error("Por favor crea un paciente primero")
    else:
        try:
            with st.spinner(f"Generando plan de alimentación personalizado usando {ai_provider}..."):
                with get_db() as session:
                    patient = session.query(Patient).filter_by(id=st.session_state.current_patient_id).first()
                    lab_values = (
                        session.query(LabValue)
                        .filter_by(patient_id=patient.id)
                        .order_by(LabValue.created_at.desc())
                        .first()
                    )

                    plan_text = generate_diet_plan(patient, lab_values, special_considerations, api_key, ai_provider)

                    new_plan = DietPlan(
                        patient_id=patient.id,
                        plan_details=plan_text,
                        special_considerations=special_considerations,
                        status="active",
                    )
                    session.add(new_plan)
                    session.flush()

                    st.session_state.plan_generated = True
                    st.session_state.current_plan = plan_text
                    st.session_state.current_plan_id = new_plan.id

                st.success(f"✅ ¡Plan generado exitosamente! (ID: {st.session_state.current_plan_id})")
                st.rerun()
        except Exception as e:
            st.error(f"Error al generar plan: {e}")

st.markdown("---")

# ══════════════════════════════════════════════
# Section: Past plans
# ══════════════════════════════════════════════
if st.session_state.patient_created and st.session_state.current_patient_id:
    st.header("📚 Planes Anteriores")

    with get_db() as session:
        past_plans = (
            session.query(DietPlan)
            .filter_by(patient_id=st.session_state.current_patient_id)
            .order_by(DietPlan.created_at.desc())
            .all()
        )
        session.expunge_all()

    if past_plans:
        st.info(f"Este paciente tiene {len(past_plans)} plan(es) guardado(s)")
        for idx, plan in enumerate(past_plans):
            with st.expander(
                f"📋 Plan #{idx + 1} - Creado el {plan.created_at.strftime('%d/%m/%Y %H:%M')}",
                expanded=(idx == 0 and len(past_plans) == 1),
            ):
                render_plan_card(plan, prefix=f"past_{idx}")
    else:
        st.info("Este paciente no tiene planes guardados aún. Genera uno en la sección de arriba.")

st.markdown("---")

# ══════════════════════════════════════════════
# Section 3: Generated plan & modifications
# ══════════════════════════════════════════════
if st.session_state.plan_generated and st.session_state.current_plan:
    st.header("3️⃣ Plan de Alimentación Generado")

    st.text_area(
        "Plan de Alimentación",
        value=st.session_state.current_plan,
        height=400,
        disabled=True,
    )

    st.download_button(
        label="📥 Descargar Plan",
        data=st.session_state.current_plan,
        file_name=f"plan_alimentacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )

    st.markdown("---")
    st.subheader("Modificar y Regenerar Plan")

    modifications = st.text_area(
        "Ingresa modificaciones o requisitos adicionales",
        placeholder="ej: Agregar más opciones de proteína, reducir carbohidratos, incluir alternativas vegetarianas",
        height=100,
    )

    if st.button("🔄 Regenerar Plan", type="secondary", disabled=not modifications):
        try:
            with st.spinner(f"Regenerando plan con modificaciones usando {ai_provider}..."):
                with get_db() as session:
                    patient = session.query(Patient).filter_by(id=st.session_state.current_patient_id).first()
                    lab_values = (
                        session.query(LabValue)
                        .filter_by(patient_id=patient.id)
                        .order_by(LabValue.created_at.desc())
                        .first()
                    )

                    modified_considerations = f"{special_considerations}\n\nModificaciones solicitadas: {modifications}"
                    new_plan_text = generate_diet_plan(patient, lab_values, modified_considerations, api_key, ai_provider)

                    existing_plan = session.query(DietPlan).filter_by(id=st.session_state.current_plan_id).first()
                    if existing_plan:
                        existing_plan.plan_details = new_plan_text
                        existing_plan.special_considerations = modified_considerations
                        existing_plan.updated_at = _utcnow()
                        st.session_state.current_plan = new_plan_text
                        st.success("✅ ¡Plan regenerado exitosamente!")
                    else:
                        st.error("No se encontró el plan para actualizar")

                st.rerun()
        except Exception as e:
            st.error(f"Error al regenerar plan: {e}")

# ── Footer ──
st.markdown("---")
st.caption("💡 Para seleccionar o crear otro paciente, usa el selector en la parte superior de la página.")
