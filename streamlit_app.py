import streamlit as st
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime, Text, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from openai import OpenAI
import anthropic

# Configuración de página
st.set_page_config(
    page_title="Asistente de Nutrición con IA",
    page_icon="🥗",
    layout="wide"
)

# Configuración de base de datos
Base = declarative_base()

# Modelos (mismos que tus modelos FastAPI)
class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    weight = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    health_conditions = Column(JSON, default=[])
    bmi = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
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
    created_at = Column(DateTime, default=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="lab_values")

class DietPlan(Base):
    __tablename__ = "diet_plans"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"))
    plan_details = Column(Text, nullable=False)
    special_considerations = Column(Text)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="diet_plans")

# Inicializar base de datos
@st.cache_resource
def init_db():
    try:
        # Check if DATABASE_URL exists in secrets
        if "DATABASE_URL" not in st.secrets:
            st.error("❌ ERROR: DATABASE_URL no está configurado en los secrets de Streamlit.")
            st.info("""
            **Para configurar DATABASE_URL:**
            1. Ve a tu app en Streamlit Cloud
            2. Click en Settings (⚙️)
            3. Click en Secrets
            4. Agrega: `DATABASE_URL = "tu_connection_string_de_supabase"`
            5. Reinicia la app
            """)
            st.stop()
            
        database_url = st.secrets["DATABASE_URL"]
        
        # Validate connection string format
        if not database_url.startswith("postgresql://"):
            st.error("❌ ERROR: DATABASE_URL debe comenzar con 'postgresql://'")
            st.stop()
        
        engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=3600)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        return Session
    
    except Exception as e:
        st.error(f"❌ ERROR al conectar con la base de datos: {str(e)}")
        st.info("""
        **Posibles soluciones:**
        1. Verifica que tu proyecto de Supabase esté activo
        2. Verifica que el DATABASE_URL sea correcto
        3. Verifica que la contraseña no tenga caracteres especiales sin escapar
        4. Intenta regenerar el DATABASE_URL en Supabase
        """)
        st.stop()

Session = init_db()

# Funciones auxiliares
def calculate_bmi(weight, height):
    """Calcular IMC desde peso (kg) y altura (cm)"""
    height_m = height / 100
    return round(weight / (height_m ** 2), 2)

def generate_diet_plan_openai(patient, lab_values, special_considerations, api_key):
    """Generar plan de dieta usando OpenAI"""
    client = OpenAI(api_key=api_key)
    
    prompt = f"""Eres un nutriólogo experto mexicano. Crea un plan de alimentación integral y personalizado para el siguiente paciente:

Información del Paciente:
- Nombre: {patient.name}
- Edad: {patient.age} años
- Género: {patient.gender}
- Peso: {patient.weight} kg
- Altura: {patient.height} cm
- IMC: {patient.bmi}
- Condiciones de Salud: {', '.join(patient.health_conditions) if patient.health_conditions else 'Ninguna'}

Resultados de Laboratorio:
- Glucosa: {lab_values.glucose if lab_values and lab_values.glucose else 'N/A'} mg/dL
- Colesterol: {lab_values.cholesterol if lab_values and lab_values.cholesterol else 'N/A'} mg/dL
- Triglicéridos: {lab_values.triglycerides if lab_values and lab_values.triglycerides else 'N/A'} mg/dL
- Hemoglobina: {lab_values.hemoglobin if lab_values and lab_values.hemoglobin else 'N/A'} g/dL

Consideraciones Especiales: {special_considerations if special_considerations else 'Ninguna'}

Por favor crea un plan detallado de 7 días que incluya:
1. Objetivos calóricos diarios
2. Distribución de macronutrientes
3. Sugerencias específicas de comidas (desayuno, comida, cena, colaciones)
4. Alimentos a evitar basados en las condiciones de salud
5. Recomendaciones de hidratación
6. Sugerencias de suplementos si es necesario
7. Consejos para el éxito

Formatea el plan de manera clara y fácil de seguir. Usa alimentos comunes en México."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2500
    )
    
    return response.choices[0].message.content

def generate_diet_plan_anthropic(patient, lab_values, special_considerations, api_key):
    """Generar plan de dieta usando Anthropic Claude"""
    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""Eres un nutriólogo experto mexicano. Crea un plan de alimentación integral y personalizado para el siguiente paciente:

Información del Paciente:
- Nombre: {patient.name}
- Edad: {patient.age} años
- Género: {patient.gender}
- Peso: {patient.weight} kg
- Altura: {patient.height} cm
- IMC: {patient.bmi}
- Condiciones de Salud: {', '.join(patient.health_conditions) if patient.health_conditions else 'Ninguna'}

Resultados de Laboratorio:
- Glucosa: {lab_values.glucose if lab_values and lab_values.glucose else 'N/A'} mg/dL
- Colesterol: {lab_values.cholesterol if lab_values and lab_values.cholesterol else 'N/A'} mg/dL
- Triglicéridos: {lab_values.triglycerides if lab_values and lab_values.triglycerides else 'N/A'} mg/dL
- Hemoglobina: {lab_values.hemoglobin if lab_values and lab_values.hemoglobin else 'N/A'} g/dL

Consideraciones Especiales: {special_considerations if special_considerations else 'Ninguna'}

Por favor crea un plan detallado de 7 días que incluya:
1. Objetivos calóricos diarios
2. Distribución de macronutrientes
3. Sugerencias específicas de comidas (desayuno, comida, cena, colaciones)
4. Alimentos a evitar basados en las condiciones de salud
5. Recomendaciones de hidratación
6. Sugerencias de suplementos si es necesario
7. Consejos para el éxito

Formatea el plan de manera clara y fácil de seguir. Usa alimentos comunes en México."""

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text

def reset_form():
    """Resetear todos los campos del formulario"""
    keys_to_delete = [
        'patient_created', 'plan_generated', 'current_patient_id', 
        'current_plan', 'current_plan_id', 'load_existing_patient',
        'patient_name', 'patient_age', 'patient_gender', 'patient_weight',
        'patient_height', 'patient_health_conditions', 'patient_glucose',
        'patient_cholesterol', 'patient_triglycerides', 'patient_hemoglobin'
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

# Aplicación Principal
st.title("🥗 Asistente de Nutrición con IA - MVP")
st.markdown("---")

# Barra lateral para configuración de API
with st.sidebar:
    st.header("⚙️ Configuración")
    
    ai_provider = st.selectbox(
        "Selecciona el Proveedor de IA",
        ["OpenAI", "Anthropic"]
    )
    
    api_key = st.text_input(
        f"Ingresa tu API Key de {ai_provider}",
        type="password",
        help=f"Obtén tu API key en {ai_provider.lower()}.com"
    )
    
    st.markdown("---")
    st.caption("Tu API key solo se usa para esta sesión y nunca se almacena.")

# Contenido principal
if not api_key:
    st.warning(f"👈 Por favor ingresa tu API key de {ai_provider} en la barra lateral para comenzar.")
    st.info("""
    **Cómo obtener una API key:**
    - **OpenAI**: Visita [platform.openai.com](https://platform.openai.com) → API Keys
    - **Anthropic**: Visita [console.anthropic.com](https://console.anthropic.com) → API Keys
    """)
    st.stop()

# Inicializar estado de sesión
if 'patient_created' not in st.session_state:
    st.session_state.patient_created = False
if 'plan_generated' not in st.session_state:
    st.session_state.plan_generated = False
if 'current_patient_id' not in st.session_state:
    st.session_state.current_patient_id = None
if 'current_plan' not in st.session_state:
    st.session_state.current_plan = None
if 'current_plan_id' not in st.session_state:
    st.session_state.current_plan_id = None
if 'load_existing_patient' not in st.session_state:
    st.session_state.load_existing_patient = False

# Sección 0: Buscar Paciente Existente
st.header("🔍 Buscar Paciente Existente")

session = Session()
all_patients = session.query(Patient).order_by(Patient.created_at.desc()).all()
session.close()

if all_patients:
    patient_options = {f"{p.name} (ID: {p.id}) - {p.age} años": p.id for p in all_patients}
    patient_options = {"-- Crear Nuevo Paciente --": None, **patient_options}
    
    selected_patient = st.selectbox(
        "Selecciona un paciente existente o crea uno nuevo",
        options=list(patient_options.keys())
    )
    
    if patient_options[selected_patient] is not None:
        if st.button("📋 Cargar Paciente Seleccionado"):
            session = Session()
            patient = session.query(Patient).filter_by(id=patient_options[selected_patient]).first()
            lab_values = session.query(LabValue).filter_by(patient_id=patient.id).order_by(LabValue.created_at.desc()).first()
            
            # Guardar datos del paciente en session_state
            st.session_state.patient_created = True
            st.session_state.current_patient_id = patient.id
            st.session_state.load_existing_patient = True
            
            # Guardar género mapeado
            gender_map_reverse = {"male": "Masculino", "female": "Femenino", "other": "Otro"}
            
            # Guardar toda la información
            st.session_state.patient_name = patient.name
            st.session_state.patient_age = patient.age
            st.session_state.patient_gender = gender_map_reverse.get(patient.gender, "Masculino")
            st.session_state.patient_weight = patient.weight
            st.session_state.patient_height = patient.height
            st.session_state.patient_health_conditions = ', '.join(patient.health_conditions) if patient.health_conditions else ''
            
            if lab_values:
                st.session_state.patient_glucose = lab_values.glucose if lab_values.glucose else 0.0
                st.session_state.patient_cholesterol = lab_values.cholesterol if lab_values.cholesterol else 0.0
                st.session_state.patient_triglycerides = lab_values.triglycerides if lab_values.triglycerides else 0.0
                st.session_state.patient_hemoglobin = lab_values.hemoglobin if lab_values.hemoglobin else 0.0
            else:
                st.session_state.patient_glucose = 0.0
                st.session_state.patient_cholesterol = 0.0
                st.session_state.patient_triglycerides = 0.0
                st.session_state.patient_hemoglobin = 0.0
            
            session.close()
            st.success(f"✅ Paciente '{patient.name}' cargado exitosamente!")
            st.rerun()
else:
    st.info("No hay pacientes en la base de datos. Crea uno nuevo abajo.")

st.markdown("---")

# Sección 1: Información del Paciente
st.header("1️⃣ Información del Paciente")

col1, col2 = st.columns(2)

# Determinar valores por defecto basados en si hay un paciente cargado
if st.session_state.load_existing_patient:
    default_name = st.session_state.get('patient_name', '')
    default_age = st.session_state.get('patient_age', 30)
    default_gender = st.session_state.get('patient_gender', 'Masculino')
    default_weight = st.session_state.get('patient_weight', 70.0)
    default_height = st.session_state.get('patient_height', 170.0)
    default_health_conditions = st.session_state.get('patient_health_conditions', '')
    default_glucose = st.session_state.get('patient_glucose', 0.0)
    default_cholesterol = st.session_state.get('patient_cholesterol', 0.0)
    default_triglycerides = st.session_state.get('patient_triglycerides', 0.0)
    default_hemoglobin = st.session_state.get('patient_hemoglobin', 0.0)
else:
    default_name = ''
    default_age = 30
    default_gender = 'Masculino'
    default_weight = 70.0
    default_height = 170.0
    default_health_conditions = ''
    default_glucose = 0.0
    default_cholesterol = 0.0
    default_triglycerides = 0.0
    default_hemoglobin = 0.0

with col1:
    name = st.text_input("Nombre *", value=default_name, disabled=st.session_state.load_existing_patient)
    age = st.number_input("Edad *", min_value=1, max_value=120, value=int(default_age), disabled=st.session_state.load_existing_patient)
    gender_index = ["Masculino", "Femenino", "Otro"].index(default_gender)
    gender = st.selectbox("Género *", ["Masculino", "Femenino", "Otro"], index=gender_index, disabled=st.session_state.load_existing_patient)

with col2:
    weight = st.number_input("Peso (kg) *", min_value=1.0, max_value=500.0, value=float(default_weight), step=0.1, disabled=st.session_state.load_existing_patient)
    height = st.number_input("Altura (cm) *", min_value=50.0, max_value=250.0, value=float(default_height), step=0.1, disabled=st.session_state.load_existing_patient)
    
    if weight and height:
        bmi = calculate_bmi(weight, height)
        st.metric("IMC", bmi)

health_conditions = st.text_input(
    "Condiciones de Salud (separadas por comas)",
    value=default_health_conditions,
    placeholder="ej: diabetes, hipertensión, enfermedad celíaca",
    disabled=st.session_state.load_existing_patient
)

st.subheader("Resultados de Laboratorio")

col3, col4 = st.columns(2)

with col3:
    glucose = st.number_input("Glucosa (mg/dL)", min_value=0.0, value=float(default_glucose), step=0.1, disabled=st.session_state.load_existing_patient)
    cholesterol = st.number_input("Colesterol (mg/dL)", min_value=0.0, value=float(default_cholesterol), step=0.1, disabled=st.session_state.load_existing_patient)

with col4:
    triglycerides = st.number_input("Triglicéridos (mg/dL)", min_value=0.0, value=float(default_triglycerides), step=0.1, disabled=st.session_state.load_existing_patient)
    hemoglobin = st.number_input("Hemoglobina (g/dL)", min_value=0.0, value=float(default_hemoglobin), step=0.1, disabled=st.session_state.load_existing_patient)

# Mostrar mensaje si el paciente fue cargado
if st.session_state.load_existing_patient:
    st.info(f"📋 Paciente cargado: {default_name} (ID: {st.session_state.current_patient_id})")

# Solo mostrar botón de crear si NO hay paciente cargado
if not st.session_state.load_existing_patient:
    if st.button("💾 Crear Paciente y Guardar Datos", type="primary", disabled=st.session_state.patient_created):
        if not name or not age or not weight or not height:
            st.error("Por favor completa todos los campos requeridos (marcados con *)")
        else:
            try:
                session = Session()
                
                # Crear paciente
                conditions_list = [c.strip() for c in health_conditions.split(',')] if health_conditions else []
                bmi_value = calculate_bmi(weight, height)
                
                # Convertir género a inglés para la base de datos
                gender_map = {"Masculino": "male", "Femenino": "female", "Otro": "other"}
                gender_db = gender_map.get(gender, "other")
                
                new_patient = Patient(
                    name=name,
                    age=int(age),
                    gender=gender_db,
                    weight=float(weight),
                    height=float(height),
                    health_conditions=conditions_list,
                    bmi=bmi_value
                )
                
                session.add(new_patient)
                session.commit()
                session.refresh(new_patient)
                
                # Crear valores de laboratorio si se proporcionaron
                if any([glucose, cholesterol, triglycerides, hemoglobin]):
                    lab_value = LabValue(
                        patient_id=new_patient.id,
                        test_date=datetime.now().strftime("%Y-%m-%d"),
                        glucose=float(glucose) if glucose > 0 else None,
                        cholesterol=float(cholesterol) if cholesterol > 0 else None,
                        triglycerides=float(triglycerides) if triglycerides > 0 else None,
                        hemoglobin=float(hemoglobin) if hemoglobin > 0 else None
                    )
                    session.add(lab_value)
                    session.commit()
                
                st.session_state.patient_created = True
                st.session_state.current_patient_id = new_patient.id
                
                st.success(f"✅ ¡Paciente creado exitosamente! (ID: {new_patient.id})")
                session.close()
                
            except Exception as e:
                st.error(f"Error al crear paciente: {str(e)}")
                if 'session' in locals():
                    session.rollback()
                    session.close()

st.markdown("---")

# Sección 2: Consideraciones Especiales y Generar Plan
st.header("2️⃣ Consideraciones Especiales y Generar Plan")

special_considerations = st.text_area(
    "Consideraciones Especiales",
    placeholder="Ingresa cualquier alergia, preferencias alimentarias, restricciones dietéticas, consideraciones culturales, etc.",
    height=100,
    key="special_considerations"
)

if st.button("🤖 Generar Plan de Alimentación", type="primary", disabled=not st.session_state.patient_created):
    if not st.session_state.patient_created:
        st.error("Por favor crea un paciente primero")
    else:
        try:
            with st.spinner(f"Generando plan de alimentación personalizado usando {ai_provider}..."):
                session = Session()
                
                # Obtener paciente y valores de laboratorio
                patient = session.query(Patient).filter_by(id=st.session_state.current_patient_id).first()
                lab_values = session.query(LabValue).filter_by(patient_id=patient.id).order_by(LabValue.created_at.desc()).first()
                
                # Generar plan
                if ai_provider == "OpenAI":
                    plan_text = generate_diet_plan_openai(patient, lab_values, special_considerations, api_key)
                else:
                    plan_text = generate_diet_plan_anthropic(patient, lab_values, special_considerations, api_key)
                
                # Guardar plan en base de datos
                new_plan = DietPlan(
                    patient_id=patient.id,
                    plan_details=plan_text,
                    special_considerations=special_considerations,
                    status="active"
                )
                
                session.add(new_plan)
                session.commit()
                session.refresh(new_plan)
                
                st.session_state.plan_generated = True
                st.session_state.current_plan = plan_text
                st.session_state.current_plan_id = new_plan.id
                
                st.success(f"✅ ¡Plan de alimentación generado exitosamente! (ID: {new_plan.id})")
                session.close()
                st.rerun()
                
        except Exception as e:
            st.error(f"Error al generar plan: {str(e)}")
            if 'session' in locals():
                session.close()

st.markdown("---")

# Sección 3: Plan Generado y Modificaciones
if st.session_state.plan_generated and st.session_state.current_plan:
    st.header("3️⃣ Plan de Alimentación Generado")
    
    # Mostrar plan
    st.text_area(
        "Plan de Alimentación",
        value=st.session_state.current_plan,
        height=400,
        disabled=True,
        key="plan_display"
    )
    
    # Botón de descarga
    st.download_button(
        label="📥 Descargar Plan",
        data=st.session_state.current_plan,
        file_name=f"plan_alimentacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )
    
    st.markdown("---")
    
    # Modificar y Regenerar
    st.subheader("Modificar y Regenerar Plan")
    
    modifications = st.text_area(
        "Ingresa modificaciones o requisitos adicionales",
        placeholder="ej: Agregar más opciones de proteína, reducir carbohidratos, incluir alternativas vegetarianas",
        height=100,
        key="modifications"
    )
    
    if st.button("🔄 Regenerar Plan", type="secondary", disabled=not modifications):
        if not modifications:
            st.error("Por favor ingresa las modificaciones")
        else:
            try:
                with st.spinner(f"Regenerando plan con modificaciones usando {ai_provider}..."):
                    session = Session()
                    
                    # Obtener paciente y valores de laboratorio
                    patient = session.query(Patient).filter_by(id=st.session_state.current_patient_id).first()
                    lab_values = session.query(LabValue).filter_by(patient_id=patient.id).order_by(LabValue.created_at.desc()).first()
                    
                    # Crear prompt modificado
                    modified_considerations = f"{special_considerations}\n\nModificaciones solicitadas: {modifications}"
                    
                    # Generar nuevo plan
                    if ai_provider == "OpenAI":
                        new_plan_text = generate_diet_plan_openai(patient, lab_values, modified_considerations, api_key)
                    else:
                        new_plan_text = generate_diet_plan_anthropic(patient, lab_values, modified_considerations, api_key)
                    
                    # Actualizar plan existente
                    existing_plan = session.query(DietPlan).filter_by(id=st.session_state.current_plan_id).first()
                    if existing_plan:
                        existing_plan.plan_details = new_plan_text
                        existing_plan.special_considerations = modified_considerations
                        existing_plan.updated_at = datetime.utcnow()
                        
                        session.commit()
                        
                        st.session_state.current_plan = new_plan_text
                        
                        st.success("✅ ¡Plan regenerado exitosamente!")
                        session.close()
                        st.rerun()
                    else:
                        st.error("No se encontró el plan para actualizar")
                        session.close()
                    
            except Exception as e:
                st.error(f"Error al regenerar plan: {str(e)}")
                if 'session' in locals():
                    session.close()

# Botón de reinicio (al final)
st.markdown("---")
if st.button("🔄 Iniciar Nuevo Paciente"):
    reset_form()
    st.rerun()
