import streamlit as st
import json
from datetime import datetime
from packaging import version
from PIL import Image

# --------------------------
# Initial Configuration
# --------------------------
st.set_page_config(
    page_title="📈 Optimizador de ROI de Marketing Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Optimized Imports with Lazy Loading
# --------------------------
def load_extras():
    """Load streamlit extras modules without caching to avoid widget conflicts"""
    try:
        from streamlit_extras.metric_cards import style_metric_cards
        from streamlit_extras.stylable_container import stylable_container
        import streamlit_extras
        
        # Version check for key parameter
        try:
            STYLABLE_CONTAINER_VERSION = version.parse(streamlit_extras.__version__)
            REQUIRES_KEY = STYLABLE_CONTAINER_VERSION >= version.parse("0.3.0")
        except:
            REQUIRES_KEY = False
            
        return {
            'extras_available': True,
            'requires_key': REQUIRES_KEY,
            'style_metric_cards': style_metric_cards,
            'stylable_container': stylable_container
        }
    except ImportError:
        return {
            'extras_available': False,
            'requires_key': False
        }

extras = load_extras()
EXTRAS_AVAILABLE = extras['extras_available']
REQUIRES_KEY = extras.get('requires_key', False)

# --------------------------
# Custom Components
# --------------------------
def get_stylable_container(styles, container_key=None):
    if not EXTRAS_AVAILABLE:
        return st.container()
    
    try:
        if REQUIRES_KEY:
            if container_key is None:
                container_key = f"container_{datetime.now().timestamp()}"
            return extras['stylable_container'](key=container_key, css_styles=styles)
        else:
            return extras['stylable_container'](css_styles=styles)
    except TypeError:
        return st.container()

def metric_card(title, value, help_text=None, delta=None):
    container = get_stylable_container(
        styles="""
            {
                background-color: #121212;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                border: 1px solid #333333;
                color: white;
                height: 120px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            h3 {
                font-size: 1rem;
                margin-bottom: 8px;
                color: #6e8efb;
            }
            .value {
                font-size: 1.8rem;
                font-weight: bold;
                margin: 5px 0;
            }
            .delta {
                font-size: 0.9rem;
            }
        """,
        container_key=f"metric_{title}"
    )
    with container:
        st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='value'>{value}</div>", unsafe_allow_html=True)
        if delta:
            st.markdown(f"<div class='delta'>{delta}</div>", unsafe_allow_html=True)
        if help_text:
            st.markdown(f"<div style='font-size:0.8rem;opacity:0.7;'>{help_text}</div>", unsafe_allow_html=True)

# --------------------------
# Custom CSS
# --------------------------
st.markdown("""
    <style>
        /* Main styles */
        .main { background-color: #000000; color: #ffffff; }
        .sidebar .sidebar-content { background-color: #121212; }
        .stMetric { background-color: #121212 !important; }
        .stDataFrame { background-color: #121212; }
        .stTabs [data-baseweb="tab"] { background-color: #121212 !important; }
        .stTabs [aria-selected="true"] { background-color: #1e1e1e !important; }
        
        /* KPI card grid */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 15px;
        }
        @media (max-width: 1200px) {
            .kpi-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        @media (max-width: 800px) {
            .kpi-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        
        /* Section styling */
        .section-header {
            border-bottom: 2px solid #6e8efb;
            padding-bottom: 8px;
            margin-top: 30px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Header Section
# --------------------------
col1, col2 = st.columns([5,1])
with col1:
    st.title("📈 Optimizador de ROI de Marketing Pro")
    st.markdown("""
        <div style="opacity: 0.9; font-size: 16px;">
        Maximiza el retorno de inversión de tu marketing con insights basados en datos de campañas multicanal en retail, 
        educación y servicios financieros.
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/3281/3281289.png", width=80)

# --------------------------
# Load Model Metadata
# --------------------------
@st.cache_data
def load_metadata():
    with open("json files/model_metadata.json") as f:
        return json.load(f)

metadata = load_metadata()

# --------------------------
# Application Overview Section
# --------------------------
st.markdown('<div class="section-header"><h2>Resumen de la Aplicación</h2></div>', unsafe_allow_html=True)

container = get_stylable_container(
    styles="""
        {
            background-color: #121212;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #333333;
        }
    """
)
with container:
    st.markdown("""
    El **Optimizador de ROI de Marketing Pro** ofrece:
    
    - **Exploración de Datos**: Visualizaciones interactivas del desempeño histórico de campañas
    - **Predicción de ROI**: Modelo de machine learning para pronosticar el retorno de inversión
    - **Análisis de Escenarios**: Herramientas "what-if" para probar diferentes asignaciones de presupuesto
    
    Desarrollado para agencias de marketing digital y equipos internos para tomar decisiones basadas en datos sobre 
    estrategias de campaña y asignación de presupuesto.
    """)

# --------------------------
# Model Information Section
# --------------------------
st.markdown('<div class="section-header"><h2>Información del Modelo</h2></div>', unsafe_allow_html=True)

# Model Details
st.markdown("### Detalles del Modelo")
col1, col2 = st.columns(2)
with col1:
    metric_card("Tipo de Modelo", 
               metadata['model_type'], 
               help_text="Algoritmo utilizado para las predicciones")
with col2:
    metric_card("Versión", 
               metadata['model_version'], 
               help_text="Versión actual del modelo")

col3, col4 = st.columns(2)
with col3:
    metric_card("Creado", 
               metadata['creation_date'][:10], 
               help_text="Fecha de creación del modelo")
with col4:
    metric_card("Variable Objetivo", 
               metadata['target'], 
               help_text="Métrica predicha")

# Performance metrics - UPDATED SECTION
st.markdown("### Métricas de Rendimiento")
test_metrics = metadata['performance_metrics']['test_improved']

# Define all possible metrics with their display names and help text
metric_definitions = {
    'RMSE': {'name': 'RMSE', 'help': 'Error Cuadrático Medio'},
    'MAE': {'name': 'MAE', 'help': 'Error Absoluto Medio'},
    'R2': {'name': 'R²', 'help': 'Coeficiente de Determinación'},
    'Explained Variance': {'name': 'Varianza Explicada', 'help': 'Varianza explicada por el modelo'},
    # Add any other metrics you want to potentially display
}

st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)

# Display each metric if it exists in the test_metrics
available_metrics = 0
for metric_key, metric_info in metric_definitions.items():
    if metric_key in test_metrics:
        available_metrics += 1
        metric_value = test_metrics[metric_key]
        # Format the value appropriately
        if isinstance(metric_value, float):
            formatted_value = f"{metric_value:.4f}"
        else:
            formatted_value = str(metric_value)
        
        metric_card(
            metric_info['name'],
            formatted_value,
            help_text=metric_info['help']
        )

st.markdown('</div>', unsafe_allow_html=True)

if available_metrics == 0:
    st.warning("No se encontraron métricas de rendimiento en los datos del modelo.")

if EXTRAS_AVAILABLE:
    extras['style_metric_cards'](border_left_color="#6e8efb", background_color="#121212")

# Top features
st.markdown("### Características Predictivas Principales")
container = get_stylable_container(
    styles="""
        {
            background-color: #121212;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #333333;
        }
    """
)
with container:
    top_features = metadata['selected_features'][:10]
    st.write("Estas características tuvieron el mayor impacto en las predicciones de ROI:")
    for i, feature in enumerate(top_features, 1):
        st.markdown(f"<div style='padding: 5px 0;'>{i}. {feature}</div>", unsafe_allow_html=True)

# --------------------------
# Methodology Section
# --------------------------
st.markdown('<div class="section-header"><h2>Metodología</h2></div>', unsafe_allow_html=True)

container = get_stylable_container(
    styles="""
        {
            background-color: #121212;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #333333;
        }
    """
)
with container:
    st.markdown("""
    1. **Recolección de Datos**: Datos históricos de campañas de múltiples canales
    2. **Preprocesamiento**: Manejo de valores faltantes, outliers e ingeniería de características
    3. **Entrenamiento del Modelo**: Regresión de Random Forest optimizada para predicción de ROI
    4. **Validación**: Pruebas rigurosas en conjuntos de datos de holdout
    5. **Despliegue**: Integrado en esta aplicación interactiva
    """)

# --------------------------
# Team/Contact Section
# --------------------------
st.markdown('<div class="section-header"><h2>Equipo & Contacto</h2></div>', unsafe_allow_html=True)

container = get_stylable_container(
    styles="""
        {
            background-color: #121212;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #333333;
        }
    """
)
with container:
    st.markdown("""
    ### Desarrollado por: Bo Kolstrup
    - Científico de Datos con experiencia en machine learning y análisis de marketing
    - Ubicación: Concepción, Chile
    - Email: bokolstrup@gmail.com
    - Teléfono: +56 9 4259 6282
    - LinkedIn: [linkedin.com/in/bo-kolstrup](https://linkedin.com/in/bo-kolstrup)
    - GitHub: [github.com/bokols](https://github.com/bokols)
    
    <div style="margin-top: 20px; font-size: 0.9em; opacity: 0.8;">
    Última actualización: {date}
    </div>
    """.format(date=datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)