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

# Modelos
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

class ExamplePlan(Base):
    __tablename__ = "example_plans"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    patient_profile = Column(Text)  # Description of patient type (e.g., "diabetic, 45 years old, sedentary")
    plan_content = Column(Text, nullable=False)  # The actual example plan
    tags = Column(JSON, default=[])  # Tags like ["diabetes", "weight-loss", "vegetarian"]
    created_at = Column(DateTime, default=datetime.utcnow)

# Inicializar base de datos
@st.cache_resource
def init_db():
    try:
        if "DATABASE_URL" not in st.secrets:
            st.error("❌ ERROR: DATABASE_URL no está configurado en los secrets de Streamlit.")
            st.stop()
            
        database_url = st.secrets["DATABASE_URL"]
        
        if not database_url.startswith("postgresql://"):
            st.error("❌ ERROR: DATABASE_URL debe comenzar con 'postgresql://'")
            st.stop()
        
        engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=3600)
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        return Session
    
    except Exception as e:
        st.error(f"❌ ERROR al conectar con la base de datos: {str(e)}")
        st.stop()

Session = init_db()

# Funciones auxiliares
def calculate_bmi(weight, height):
    """Calcular IMC desde peso (kg) y altura (cm)"""
    height_m = height / 100
    return round(weight / (height_m ** 2), 2)

def find_relevant_examples(patient, lab_values, special_considerations, top_k=2):
    """
    Find relevant example plans based on patient conditions
    Simple keyword matching approach
    """
    session = Session()
    all_examples = session.query(ExamplePlan).all()
    session.close()
    
    if not all_examples:
        return []
    
    # Create search keywords from patient data
    search_keywords = set()
    
    # Add health conditions
    if patient.health_conditions:
        for condition in patient.health_conditions:
            search_keywords.add(condition.lower().strip())
    
    # Add special considerations
    if special_considerations:
        for word in special_considerations.lower().split():
            if len(word) > 3:  # Only meaningful words
                search_keywords.add(word.strip())
    
    # Add age range keywords
    if patient.age < 18:
        search_keywords.add("adolescente")
        search_keywords.add("joven")
    elif patient.age < 30:
        search_keywords.add("adulto joven")
    elif patient.age < 60:
        search_keywords.add("adulto")
    else:
        search_keywords.add("adulto mayor")
        search_keywords.add("tercera edad")
    
    # Add BMI category
    bmi = patient.bmi
    if bmi < 18.5:
        search_keywords.add("bajo peso")
    elif bmi < 25:
        search_keywords.add("peso normal")
    elif bmi < 30:
        search_keywords.add("sobrepeso")
    else:
        search_keywords.add("obesidad")
    
    # Score each example
    scored_examples = []
    for example in all_examples:
        score = 0
        
        # Check tags
        if example.tags:
            for tag in example.tags:
                if tag.lower() in search_keywords:
                    score += 3  # Tags are more important
        
        # Check patient profile
        if example.patient_profile:
            profile_lower = example.patient_profile.lower()
            for keyword in search_keywords:
                if keyword in profile_lower:
                    score += 2
        
        # Check plan content (less weight)
        if example.plan_content:
            content_lower = example.plan_content.lower()
            for keyword in search_keywords:
                if keyword in content_lower:
                    score += 1
        
        if score > 0:
            scored_examples.append((score, example))
    
    # Sort by score and return top_k
    scored_examples.sort(reverse=True, key=lambda x: x[0])
    return [example for score, example in scored_examples[:top_k]]

def generate_diet_plan_openai(patient, lab_values, special_considerations, api_key):
    """Generar plan de dieta usando OpenAI con RAG"""
    client = OpenAI(api_key=api_key)
    
    # Get relevant example plans
    relevant_examples = find_relevant_examples(patient, lab_values, special_considerations)
    
    # Build examples section
    examples_text = ""
    if relevant_examples:
        examples_text = "\n\nEJEMPLOS DE REFERENCIA (usa estos como guía de estilo y formato):\n\n"
        for idx, example in enumerate(relevant_examples, 1):
            examples_text += f"--- EJEMPLO {idx} ---\n"
            examples_text += f"Perfil del paciente: {example.patient_profile}\n"
            examples_text += f"Plan:\n{example.plan_content}\n\n"
    
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
{examples_text}
IMPORTANTE: Usa los ejemplos de referencia como guía para el estilo, formato y estructura del plan. Adapta el contenido específicamente para este paciente, pero mantén un estilo similar.

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
    """Generar plan de dieta usando Anthropic Claude con RAG"""
    client = anthropic.Anthropic(api_key=api_key)
    
    # Get relevant example plans
    relevant_examples = find_relevant_examples(patient, lab_values, special_considerations)
    
    # Build examples section
    examples_text = ""
    if relevant_examples:
        examples_text = "\n\nEJEMPLOS DE REFERENCIA (usa estos como guía de estilo y formato):\n\n"
        for idx, example in enumerate(relevant_examples, 1):
            examples_text += f"--- EJEMPLO {idx} ---\n"
            examples_text += f"Perfil del paciente: {example.patient_profile}\n"
            examples_text += f"Plan:\n{example.plan_content}\n\n"
    
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
{examples_text}
IMPORTANTE: Usa los ejemplos de referencia como guía para el estilo, formato y estructura del plan. Adapta el contenido específicamente para este paciente, pero mantén un estilo similar.

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

# Inicializar estado de sesión ANTES de todo
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
if 'show_add_example' not in st.session_state:
    st.session_state.show_add_example = False

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

# Nueva Sección: Gestión de Planes de Ejemplo (en el sidebar)
with st.sidebar:
    st.markdown("---")
    st.subheader("📚 Planes de Ejemplo")
    
    if st.button("➕ Agregar Plan de Ejemplo"):
        st.session_state.show_add_example = True
    
    session = Session()
    example_count = session.query(ExamplePlan).count()
    session.close()
    
    st.caption(f"Tienes {example_count} plan(es) de ejemplo guardado(s)")

# Modal para agregar ejemplo (fuera del sidebar)
# Modal para agregar ejemplo (fuera del sidebar)
if st.session_state.get('show_add_example', False):
    st.markdown("---")
    st.header("➕ Agregar Plan de Ejemplo")
    
    with st.form("add_example_form"):
        example_title = st.text_input("Título del Plan *", placeholder="ej: Plan para Diabético Tipo 2")
        
        example_profile = st.text_area(
            "Perfil del Paciente *",
            placeholder="ej: Hombre de 55 años, diabético tipo 2, sedentario, IMC 28, necesita perder peso",
            height=100
        )
        
        example_tags_input = st.text_input(
            "Etiquetas (separadas por comas) *",
            placeholder="ej: diabetes, sobrepeso, sedentario, hipertensión"
        )
        
        # File upload option
        st.markdown("**Opción 1: Subir Archivo**")
        uploaded_file = st.file_uploader(
            "Sube un archivo con el plan de ejemplo",
            type=['txt', 'md', 'docx', 'pdf'],
            help="Formatos soportados: .txt, .md, .docx, .pdf"
        )
        
        st.markdown("**O**")
        st.markdown("**Opción 2: Pegar Texto Directamente**")
        
        example_content = st.text_area(
            "Contenido del Plan",
            placeholder="Pega aquí un plan de ejemplo completo que quieras usar como referencia...",
            height=400
        )
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("💾 Guardar Ejemplo", type="primary")
        with col2:
            cancelled = st.form_submit_button("❌ Cancelar")
        
        if submitted:
            # Extract content from file or text area
            final_content = ""
            
            if uploaded_file is not None:
                # Process uploaded file
                try:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_extension in ['txt', 'md']:
                        # Read text files directly
                        final_content = uploaded_file.read().decode('utf-8')
                        
                    elif file_extension == 'docx':
                        # Read Word documents
                        import docx
                        doc = docx.Document(uploaded_file)
                        final_content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                        
                    elif file_extension == 'pdf':
                        # Read PDF files
                        import PyPDF2
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        final_content = ''
                        for page in pdf_reader.pages:
                            final_content += page.extract_text() + '\n'
                    
                    st.info(f"✅ Archivo '{uploaded_file.name}' procesado exitosamente")
                    
                except Exception as e:
                    st.error(f"Error al procesar archivo: {str(e)}")
                    final_content = ""
                    
            elif example_content:
                # Use pasted text
                final_content = example_content
            
            # Validate all fields
            if not example_title or not example_profile or not example_tags_input:
                st.error("Por favor completa el título, perfil del paciente y etiquetas")
            elif not final_content:
                st.error("Por favor sube un archivo O pega el contenido del plan")
            else:
                try:
                    session = Session()
                    
                    # Parse tags
                    tags_list = [tag.strip() for tag in example_tags_input.split(',')]
                    
                    new_example = ExamplePlan(
                        title=example_title,
                        patient_profile=example_profile,
                        plan_content=final_content,
                        tags=tags_list
                    )
                    
                    session.add(new_example)
                    session.commit()
                    
                    st.success("✅ Plan de ejemplo guardado exitosamente!")
                    st.session_state.show_add_example = False
                    session.close()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error al guardar ejemplo: {str(e)}")
                    if 'session' in locals():
                        session.rollback()
                        session.close()
        
        if cancelled:
            st.session_state.show_add_example = False
            st.rerun()
    
    st.markdown("---")
    
    # Show existing examples
    st.subheader("📋 Planes de Ejemplo Existentes")
    
    session = Session()
    all_examples = session.query(ExamplePlan).order_by(ExamplePlan.created_at.desc()).all()
    session.close()
    
    if all_examples:
        for example in all_examples:
            with st.expander(f"📄 {example.title}", expanded=False):
                st.markdown(f"**ID:** {example.id}")
                st.markdown(f"**Perfil:** {example.patient_profile}")
                st.markdown(f"**Etiquetas:** {', '.join(example.tags) if example.tags else 'Sin etiquetas'}")
                st.markdown(f"**Creado:** {example.created_at.strftime('%d/%m/%Y %H:%M')}")
                
                st.text_area(
                    "Contenido",
                    value=example.content,
                    height=200,
                    disabled=True,
                    key=f"example_content_{example.id}"
                )
                
                if st.button(f"🗑️ Eliminar", key=f"delete_example_{example.id}"):
                    try:
                        session = Session()
                        example_to_delete = session.query(ExamplePlan).filter_by(id=example.id).first()
                        if example_to_delete:
                            session.delete(example_to_delete)
                            session.commit()
                            st.success("Ejemplo eliminado")
                            session.close()
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        if 'session' in locals():
                            session.rollback()
                            session.close()
    else:
        st.info("No hay planes de ejemplo. Agrega algunos para mejorar la calidad de los planes generados.")

# Sección 0: Buscar Paciente Existente
st.header("🔍 Buscar Paciente Existente")

session = Session()
all_patients = session.query(Patient).order_by(Patient.created_at.desc()).all()
session.close()

if all_patients:
    patient_options = {"-- Crear Nuevo Paciente --": None}
    for p in all_patients:
        patient_options[f"{p.name} (ID: {p.id}) - {p.age} años"] = p.id
    
    # Determine default selection
    default_index = 0
    if st.session_state.current_patient_id:
        for idx, (label, pid) in enumerate(patient_options.items()):
            if pid == st.session_state.current_patient_id:
                default_index = idx
                break
    
    selected_patient = st.selectbox(
        "Selecciona un paciente existente o crea uno nuevo",
        options=list(patient_options.keys()),
        index=default_index,
        key="patient_selector"
    )
    
    selected_patient_id = patient_options[selected_patient]
    
    # Check if selection changed
    if selected_patient_id != st.session_state.current_patient_id:
        if selected_patient_id is not None:
            # Load patient data
            session = Session()
            patient = session.query(Patient).filter_by(id=selected_patient_id).first()
            lab_values = session.query(LabValue).filter_by(patient_id=patient.id).order_by(LabValue.created_at.desc()).first()
            
            if patient:
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
                st.rerun()
        else:
            # "Crear Nuevo Paciente" was selected
            if st.session_state.load_existing_patient:
                reset_form()
                st.rerun()
    
    # Show current status
    if st.session_state.load_existing_patient and st.session_state.current_patient_id:
        st.success(f"📋 Paciente cargado: {st.session_state.get('patient_name', 'Desconocido')} (ID: {st.session_state.current_patient_id})")
    else:
        st.info("📝 Modo: Crear nuevo paciente")
        
else:
    st.info("No hay pacientes en la base de datos. Crea uno nuevo abajo.")

st.markdown("---")

# Sección 1: Información del Paciente
st.header("1️⃣ Información del Paciente")

# Get values from session state or use defaults
name_value = st.session_state.get('patient_name', '')
age_value = st.session_state.get('patient_age', 30)
gender_value = st.session_state.get('patient_gender', 'Masculino')
weight_value = st.session_state.get('patient_weight', 70.0)
height_value = st.session_state.get('patient_height', 170.0)
health_conditions_value = st.session_state.get('patient_health_conditions', '')
glucose_value = st.session_state.get('patient_glucose', 0.0)
cholesterol_value = st.session_state.get('patient_cholesterol', 0.0)
triglycerides_value = st.session_state.get('patient_triglycerides', 0.0)
hemoglobin_value = st.session_state.get('patient_hemoglobin', 0.0)

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Nombre *", value=name_value)
    age = st.number_input("Edad *", min_value=1, max_value=120, value=int(age_value))
    gender_index = ["Masculino", "Femenino", "Otro"].index(gender_value)
    gender = st.selectbox("Género *", ["Masculino", "Femenino", "Otro"], index=gender_index)

with col2:
    weight = st.number_input("Peso (kg) *", min_value=1.0, max_value=500.0, value=float(weight_value), step=0.1)
    height = st.number_input("Altura (cm) *", min_value=50.0, max_value=250.0, value=float(height_value), step=0.1)
    
    if weight and height:
        bmi = calculate_bmi(weight, height)
        st.metric("IMC", bmi)

health_conditions = st.text_input(
    "Condiciones de Salud (separadas por comas)",
    value=health_conditions_value,
    placeholder="ej: diabetes, hipertensión, enfermedad celíaca"
)

st.subheader("Resultados de Laboratorio")

col3, col4 = st.columns(2)

with col3:
    glucose = st.number_input("Glucosa (mg/dL)", min_value=0.0, value=float(glucose_value), step=0.1)
    cholesterol = st.number_input("Colesterol (mg/dL)", min_value=0.0, value=float(cholesterol_value), step=0.1)

with col4:
    triglycerides = st.number_input("Triglicéridos (mg/dL)", min_value=0.0, value=float(triglycerides_value), step=0.1)
    hemoglobin = st.number_input("Hemoglobina (g/dL)", min_value=0.0, value=float(hemoglobin_value), step=0.1)

# Botones según el estado
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    # Botón para CREAR nuevo paciente
    if not st.session_state.load_existing_patient:
        if st.button("💾 Crear Paciente y Guardar Datos", type="primary", disabled=st.session_state.patient_created):
            if not name or not age or not weight or not height:
                st.error("Por favor completa todos los campos requeridos (marcados con *)")
            else:
                try:
                    session = Session()
                    
                    conditions_list = [c.strip() for c in health_conditions.split(',')] if health_conditions else []
                    bmi_value = calculate_bmi(weight, height)
                    
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

with col_btn2:
    # Botón para ACTUALIZAR paciente existente
    if st.session_state.load_existing_patient:
        if st.button("🔄 Actualizar Datos del Paciente", type="primary"):
            if not name or not age or not weight or not height:
                st.error("Por favor completa todos los campos requeridos (marcados con *)")
            else:
                try:
                    session = Session()
                    
                    patient = session.query(Patient).filter_by(id=st.session_state.current_patient_id).first()
                    
                    if patient:
                        conditions_list = [c.strip() for c in health_conditions.split(',')] if health_conditions else []
                        bmi_value = calculate_bmi(weight, height)
                        
                        gender_map = {"Masculino": "male", "Femenino": "female", "Otro": "other"}
                        gender_db = gender_map.get(gender, "other")
                        
                        patient.name = name
                        patient.age = int(age)
                        patient.gender = gender_db
                        patient.weight = float(weight)
                        patient.height = float(height)
                        patient.health_conditions = conditions_list
                        patient.bmi = bmi_value
                        patient.updated_at = datetime.utcnow()
                        
                        lab_values = session.query(LabValue).filter_by(patient_id=patient.id).order_by(LabValue.created_at.desc()).first()
                        
                        if any([glucose, cholesterol, triglycerides, hemoglobin]):
                            if lab_values:
                                lab_values.glucose = float(glucose) if glucose > 0 else None
                                lab_values.cholesterol = float(cholesterol) if cholesterol > 0 else None
                                lab_values.triglycerides = float(triglycerides) if triglycerides > 0 else None
                                lab_values.hemoglobin = float(hemoglobin) if hemoglobin > 0 else None
                                lab_values.test_date = datetime.now().strftime("%Y-%m-%d")
                            else:
                                new_lab_value = LabValue(
                                    patient_id=patient.id,
                                    test_date=datetime.now().strftime("%Y-%m-%d"),
                                    glucose=float(glucose) if glucose > 0 else None,
                                    cholesterol=float(cholesterol) if cholesterol > 0 else None,
                                    triglycerides=float(triglycerides) if triglycerides > 0 else None,
                                    hemoglobin=float(hemoglobin) if hemoglobin > 0 else None
                                )
                                session.add(new_lab_value)
                        
                        session.commit()
                        
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
                        
                        st.success(f"✅ ¡Datos del paciente actualizados exitosamente! (ID: {patient.id})")
                        session.close()
                        
                    else:
                        st.error("No se encontró el paciente")
                        session.close()
                    
                except Exception as e:
                    st.error(f"Error al actualizar paciente: {str(e)}")
                    if 'session' in locals():
                        session.rollback()
                        session.close()

st.markdown("---")

# Sección 2: Consideraciones Especiales y Generar Plan
st.header("2️⃣ Consideraciones Especiales y Generar Plan")

special_considerations = st.text_area(
    "Consideraciones Especiales",
    placeholder="Ingresa cualquier alergia, preferencias alimentarias, restricciones dietéticas, consideraciones culturales, etc.",
    height=100
)

if st.button("🤖 Generar Plan de Alimentación", type="primary", disabled=not st.session_state.patient_created):
    if not st.session_state.patient_created:
        st.error("Por favor crea un paciente primero")
    else:
        try:
            with st.spinner(f"Generando plan de alimentación personalizado usando {ai_provider}..."):
                session = Session()
                
                patient = session.query(Patient).filter_by(id=st.session_state.current_patient_id).first()
                lab_values = session.query(LabValue).filter_by(patient_id=patient.id).order_by(LabValue.created_at.desc()).first()
                
                if ai_provider == "OpenAI":
                    plan_text = generate_diet_plan_openai(patient, lab_values, special_considerations, api_key)
                else:
                    plan_text = generate_diet_plan_anthropic(patient, lab_values, special_considerations, api_key)
                
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
st.markdown("---")

# Nueva Sección: Ver Planes Anteriores
if st.session_state.patient_created and st.session_state.current_patient_id:
    st.header("📚 Planes Anteriores")
    
    session = Session()
    past_plans = session.query(DietPlan).filter_by(
        patient_id=st.session_state.current_patient_id
    ).order_by(DietPlan.created_at.desc()).all()
    session.close()
    
    if past_plans:
        st.info(f"Este paciente tiene {len(past_plans)} plan(es) guardado(s)")
        
        # Create tabs for each plan
        if len(past_plans) == 1:
            # If only one plan, just show it directly
            plan = past_plans[0]
            with st.expander(f"📋 Plan creado el {plan.created_at.strftime('%d/%m/%Y %H:%M')}", expanded=False):
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
                    key=f"past_plan_{plan.id}"
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        label="📥 Descargar",
                        data=plan.plan_details,
                        file_name=f"plan_{plan.id}_{plan.created_at.strftime('%Y%m%d')}.txt",
                        mime="text/plain",
                        key=f"download_past_{plan.id}"
                    )
                with col2:
                    if st.button("🔄 Cargar para Editar", key=f"load_plan_{plan.id}"):
                        st.session_state.current_plan = plan.plan_details
                        st.session_state.current_plan_id = plan.id
                        st.session_state.plan_generated = True
                        st.success("Plan cargado para editar")
                        st.rerun()
                with col3:
                    if st.button("🗑️ Eliminar Plan", key=f"delete_plan_{plan.id}", type="secondary"):
                        try:
                            session = Session()
                            plan_to_delete = session.query(DietPlan).filter_by(id=plan.id).first()
                            if plan_to_delete:
                                session.delete(plan_to_delete)
                                session.commit()
                                st.success("Plan eliminado exitosamente")
                                session.close()
                                st.rerun()
                            else:
                                st.error("Plan no encontrado")
                                session.close()
                        except Exception as e:
                            st.error(f"Error al eliminar plan: {str(e)}")
                            if 'session' in locals():
                                session.rollback()
                                session.close()
        else:
            # Multiple plans - use expanders
            for idx, plan in enumerate(past_plans):
                with st.expander(
                    f"📋 Plan #{idx + 1} - Creado el {plan.created_at.strftime('%d/%m/%Y %H:%M')}", 
                    expanded=(idx == 0)  # First plan expanded by default
                ):
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
                        key=f"past_plan_{plan.id}"
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.download_button(
                            label="📥 Descargar",
                            data=plan.plan_details,
                            file_name=f"plan_{plan.id}_{plan.created_at.strftime('%Y%m%d')}.txt",
                            mime="text/plain",
                            key=f"download_past_{plan.id}"
                        )
                    with col2:
                        if st.button("🔄 Cargar para Editar", key=f"load_plan_{plan.id}"):
                            st.session_state.current_plan = plan.plan_details
                            st.session_state.current_plan_id = plan.id
                            st.session_state.plan_generated = True
                            st.success("Plan cargado para editar")
                            st.rerun()
                    with col3:
                        if st.button("🗑️ Eliminar Plan", key=f"delete_plan_{plan.id}", type="secondary"):
                            try:
                                session = Session()
                                plan_to_delete = session.query(DietPlan).filter_by(id=plan.id).first()
                                if plan_to_delete:
                                    session.delete(plan_to_delete)
                                    session.commit()
                                    st.success("Plan eliminado exitosamente")
                                    session.close()
                                    st.rerun()
                                else:
                                    st.error("Plan no encontrado")
                                    session.close()
                            except Exception as e:
                                st.error(f"Error al eliminar plan: {str(e)}")
                                if 'session' in locals():
                                    session.rollback()
                                    session.close()
    else:
        st.info("Este paciente no tiene planes guardados aún. Genera uno en la sección de abajo.")

st.markdown("---")

# Sección 3: Plan Generado y Modificaciones
if st.session_state.plan_generated and st.session_state.current_plan:
    st.header("3️⃣ Plan de Alimentación Generado")
    
    st.text_area(
        "Plan de Alimentación",
        value=st.session_state.current_plan,
        height=400,
        disabled=True
    )
    
    st.download_button(
        label="📥 Descargar Plan",
        data=st.session_state.current_plan,
        file_name=f"plan_alimentacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )
    
    st.markdown("---")
    
    st.subheader("Modificar y Regenerar Plan")
    
    modifications = st.text_area(
        "Ingresa modificaciones o requisitos adicionales",
        placeholder="ej: Agregar más opciones de proteína, reducir carbohidratos, incluir alternativas vegetarianas",
        height=100
    )
    
    if st.button("🔄 Regenerar Plan", type="secondary", disabled=not modifications):
        if not modifications:
            st.error("Por favor ingresa las modificaciones")
        else:
            try:
                with st.spinner(f"Regenerando plan con modificaciones usando {ai_provider}..."):
                    session = Session()
                    
                    patient = session.query(Patient).filter_by(id=st.session_state.current_patient_id).first()
                    lab_values = session.query(LabValue).filter_by(patient_id=patient.id).order_by(LabValue.created_at.desc()).first()
                    
                    modified_considerations = f"{special_considerations}\n\nModificaciones solicitadas: {modifications}"
                    
                    if ai_provider == "OpenAI":
                        new_plan_text = generate_diet_plan_openai(patient, lab_values, modified_considerations, api_key)
                    else:
                        new_plan_text = generate_diet_plan_anthropic(patient, lab_values, modified_considerations, api_key)
                    
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

# Botón de reinicio
st.markdown("---")
if st.button("🔄 Iniciar Nuevo Paciente"):
    reset_form()
    st.rerun()
