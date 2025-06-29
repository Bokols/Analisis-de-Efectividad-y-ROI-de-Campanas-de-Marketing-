"""
PREDICTOR DE ROI DE MARKETING PRO
===========================

Una aplicación Streamlit que predice el Retorno de la Inversión (ROI) para campañas de marketing
utilizando aprendizaje automático. Esta herramienta ayuda a los especialistas en marketing a optimizar sus presupuestos proporcionando
información basada en datos sobre el rendimiento esperado de las campañas.

Características principales:
- Predicción de ROI basada en parámetros de campaña
- Análisis de escenarios para probar diferentes asignaciones de presupuesto
- Cálculo de métricas de rendimiento (tasas de conversión, clics, engagement)
- Visualizaciones interactivas
- Diseño responsive para diferentes tamaños de pantalla

Tecnologías utilizadas:
- Python 3.9+
- Streamlit para la interfaz web
- Scikit-learn para aprendizaje automático
- Plotly para visualizaciones
- Pandas para manipulación de datos
"""

# --------------------------
# Initial Configuration
# --------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from datetime import datetime
from packaging import version
from PIL import Image
import sklearn

# Configure Streamlit page settings
st.set_page_config(
    page_title="🔮 Predictor de ROI de Marketing Pro",  # Page title shown in browser tab
    page_icon="📈",                              # Favicon emoji
    layout="wide",                               # Use full page width
    initial_sidebar_state="expanded"             # Show sidebar by default
)

# --------------------------
# Optimized Imports with Lazy Loading
# --------------------------
def load_extras():
    """
    Dynamically loads Streamlit extras modules to avoid widget conflicts.
    Implements lazy loading for better performance.
    
    Returns:
        dict: Dictionary containing loaded modules and version info
    """
    try:
        from streamlit_extras.metric_cards import style_metric_cards
        from streamlit_extras.dataframe_explorer import dataframe_explorer
        from streamlit_extras.stylable_container import stylable_container
        import streamlit_extras
        
        # Version check for key parameter (required in newer versions)
        try:
            STYLABLE_CONTAINER_VERSION = version.parse(streamlit_extras.__version__)
            REQUIRES_KEY = STYLABLE_CONTAINER_VERSION >= version.parse("0.3.0")
        except:
            REQUIRES_KEY = False
            
        return {
            'extras_available': True,
            'requires_key': REQUIRES_KEY,
            'style_metric_cards': style_metric_cards,
            'dataframe_explorer': dataframe_explorer,
            'stylable_container': stylable_container
        }
    except ImportError:
        # Graceful fallback if extras aren't installed
        return {
            'extras_available': False,
            'requires_key': False
        }

# Load extras once and store in global variable
extras = load_extras()
EXTRAS_AVAILABLE = extras['extras_available']  # Flag indicating if extras are available
REQUIRES_KEY = extras.get('requires_key', False)  # Version-specific behavior flag

# --------------------------
# Custom UI Components
# --------------------------
def get_stylable_container(styles, container_key=None):
    """
    Creates a stylable container with fallback to standard container if extras not available.
    
    Args:
        styles (str): CSS styles to apply to the container
        container_key (str, optional): Unique key for the container (required in newer versions)
    
    Returns:
        streamlit.container: Styled container component
    """
    if not EXTRAS_AVAILABLE:
        return st.container()  # Fallback to standard container
    
    try:
        if REQUIRES_KEY:
            if container_key is None:
                # Generate unique key if not provided
                container_key = f"container_{datetime.now().timestamp()}"
            return extras['stylable_container'](key=container_key, css_styles=styles)
        else:
            return extras['stylable_container'](css_styles=styles)
    except TypeError:
        return st.container()  # Fallback on error

def metric_card(title, value, help_text=None, delta=None):
    """
    Creates a styled metric card component with consistent formatting.
    
    Args:
        title (str): Metric title
        value (str/number): Main value to display
        help_text (str, optional): Descriptive text shown below value
        delta (str/number, optional): Change indicator (e.g., +5%)
    """
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
        container_key=f"metric_{title}"  # Unique key for each card
    )
    
    with container:
        st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='value'>{value}</div>", unsafe_allow_html=True)
        if delta:
            st.markdown(f"<div class='delta'>{delta}</div>", unsafe_allow_html=True)
        if help_text:
            st.markdown(f"<div style='font-size:0.8rem;opacity:0.7;'>{help_text}</div>", 
                        unsafe_allow_html=True)

# --------------------------
# Custom CSS Styling
# --------------------------
st.markdown("""
    <style>
        /* Main page styling */
        .main { 
            background-color: #000000; 
            color: #ffffff; 
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content { 
            background-color: #121212; 
        }
        
        /* Metric card styling */
        .stMetric { 
            background-color: #121212 !important; 
        }
        
        /* Dataframe styling */
        .stDataFrame { 
            background-color: #121212; 
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab"] { 
            background-color: #121212 !important; 
        }
        .stTabs [aria-selected="true"] { 
            background-color: #1e1e1e !important; 
        }
        
        /* Form styling */
        .stForm {
            background-color: #121212;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #333333;
        }
        
        /* Button styling with hover effects */
        .stButton>button {
            background-color: #6e8efb;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #5a7df4;
            transform: translateY(-2px);
        }
        
        /* Input field styling */
        .stNumberInput>div>div>input, 
        .stTextInput>div>div>input,
        .stSelectbox>div>div>select {
            background-color: #1e1e1e !important;
            color: white !important;
            border: 1px solid #333333 !important;
        }
        
        /* Slider styling */
        .stSlider>div>div>div>div {
            background-color: #6e8efb !important;
        }
        
        /* Responsive KPI card grid */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
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
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Header Section
# --------------------------
col1, col2 = st.columns([5,1])  # Create two columns (5:1 ratio)
with col1:
    st.title("🔮 Predictor de ROI de Marketing Pro")
    st.markdown("""
        <div style="opacity: 0.9; font-size: 16px;">
        Predice el Retorno de la Inversión (ROI) para tus campañas de marketing basado en datos históricos
        y modelos de aprendizaje automático. Optimiza tu gasto en marketing con información basada en datos.
        </div>
    """, unsafe_allow_html=True)
with col2:
    # Display marketing ROI image
    st.image("https://cdn-icons-png.flaticon.com/512/3281/3281289.png", width=80)

# --------------------------
# Session State Initialization
# --------------------------
if 'roi_prediction' not in st.session_state:
    st.session_state.roi_prediction = None  # Stores the ROI prediction result
    
if 'input_data' not in st.session_state:
    st.session_state.input_data = None  # Stores the input data used for prediction

# --------------------------
# Model and Data Loading
# --------------------------
@st.cache_resource  # Cache the model loading to improve performance
def load_model():
    """
    Loads the pre-trained machine learning model and preprocessor.
    Includes workaround for scikit-learn version compatibility issues.
    
    Returns:
        tuple: (model, preprocessor) - The trained model and preprocessing pipeline
    """
    # Workaround for scikit-learn version compatibility
    import sklearn.compose._column_transformer
    if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
        class _RemainderColsList(list):
            pass
        sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
    
    try:
        model = joblib.load("models/marketing_roi_predictor_rf.pkl")
        preprocessor = joblib.load("models/preprocessor.pkl")
        return model, preprocessor
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.stop()  # Stop execution if model can't be loaded

# Load model and preprocessor
model, preprocessor = load_model()

@st.cache_data  # Cache the data loading to improve performance
def load_data():
    """
    Loads the historical marketing data used for reference and calculations.
    
    Returns:
        DataFrame: Pandas DataFrame containing marketing campaign data
    """
    # Updated to read Parquet file with absolute path
    df = pd.read_parquet("data/marketing_data_for_powerbi.parquet")
    return df

# Load marketing data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error al cargar los datos: {str(e)}")
    st.stop()  # Stop execution if data can't be loaded

# --------------------------
# Metric Calculation Functions
# --------------------------
def calculate_conversion_rate(channel, campaign_type, target_audience, product_type, 
                            creative_rating, duration, seasonality):
    """
    Calculates estimated conversion rate based on campaign parameters.
    
    Args:
        channel (str): Marketing channel (e.g., 'Social Media', 'Email')
        campaign_type (str): Type of campaign (e.g., 'Brand Awareness')
        target_audience (str): Target demographic group
        product_type (str): Type of product being promoted
        creative_rating (int): Quality rating of creative assets (1-10)
        duration (int): Campaign duration in days
        seasonality (str): Seasonality factor ('Low', 'Medium', 'High', 'Peak')
    
    Returns:
        float: Estimated conversion rate (0-1)
    """
    # Base conversion rates by channel and campaign type
    channel_factors = {
        'Social Media': {'Brand Awareness': 0.03, 'Product Launch': 0.05, 'Promotional': 0.04},
        'Email': {'Brand Awareness': 0.02, 'Product Launch': 0.06, 'Promotional': 0.08},
        'TV': {'Brand Awareness': 0.01, 'Product Launch': 0.03, 'Promotional': 0.02},
        'Radio': {'Brand Awareness': 0.008, 'Product Launch': 0.015, 'Promotional': 0.01},
        'Search': {'Brand Awareness': 0.04, 'Product Launch': 0.07, 'Promotional': 0.09}
    }
    
    # Audience multipliers
    audience_factors = {
        'Young Adults': 1.2,
        'Families': 1.1,
        'Professionals': 1.3,
        'Seniors': 0.9
    }
    
    # Product type multipliers
    product_factors = {
        'Electronics': 1.4,
        'Clothing': 1.1,
        'Food': 1.0,
        'Services': 1.2
    }
    
    # Duration factor (longer campaigns perform better)
    duration_factor = min(1 + (duration - 7) / 21, 1.5)  # Cap at 1.5x
    
    # Seasonality factors
    seasonality_factors = {
        'Low': 0.7,
        'Medium': 1.0,
        'High': 1.3,
        'Peak': 1.6
    }
    
    try:
        base_rate = channel_factors[channel][campaign_type]
    except KeyError:
        # Fallback to median if combination not found
        base_rate = df['Conversion_Rate'].median()
    
    # Calculate final rate with all factors
    rate = (base_rate * 
            audience_factors.get(target_audience, 1.0) * 
            product_factors.get(product_type, 1.0) * 
            duration_factor * 
            seasonality_factors.get(seasonality, 1.0))
    
    # Apply creative quality factor
    creative_factor = 1 + (creative_rating - 5) * 0.03  # 3% change per point from baseline 5
    return min(rate * creative_factor, 0.5)  # Cap at 50% conversion rate

def calculate_engagement_score(channel, duration, budget, seasonality, 
                             campaign_type, creative_rating):
    """
    Calculates estimated engagement score based on campaign parameters.
    
    Args:
        channel (str): Marketing channel
        duration (int): Campaign duration in days
        budget (float): Campaign budget in CLP
        seasonality (str): Seasonality factor
        campaign_type (str): Type of campaign
        creative_rating (int): Quality rating of creative assets (1-10)
    
    Returns:
        float: Estimated engagement score (0-100)
    """
    # Base engagement scores by channel and campaign type
    base_scores = {
        'Social Media': {
            'Brand Awareness': 80,
            'Product Launch': 70,
            'Promotional': 65
        },
        'Email': {
            'Brand Awareness': 50,
            'Product Launch': 60,
            'Promotional': 70
        },
        'TV': {
            'Brand Awareness': 60,
            'Product Launch': 50,
            'Promotional': 40
        },
        'Radio': {
            'Brand Awareness': 45,
            'Product Launch': 40,
            'Promotional': 35
        },
        'Search': {
            'Brand Awareness': 60,
            'Product Launch': 75,
            'Promotional': 80
        }
    }
    
    # Seasonality multipliers
    seasonality_factors = {
        'Low': 0.8,
        'Medium': 1.0,
        'High': 1.3,
        'Peak': 1.7
    }
    
    # Calculate various factors
    creative_factor = 1 + (creative_rating - 5) * 0.05  # 5% change per point from baseline 5
    budget_factor = np.log10(budget / 10000 + 1) / 2  # Logarithmic scaling of budget impact
    duration_factor = min(1 + (duration / 30), 1.5)  # Cap at 1.5x for long durations
    
    try:
        base_score = base_scores[channel][campaign_type]
    except KeyError:
        base_score = 50  # Default score if combination not found
    
    # Calculate final score with all factors
    score = (base_score * 
             budget_factor * 
             duration_factor * 
             seasonality_factors.get(seasonality, 1.0) * 
             creative_factor)
    
    return min(score, 100)  # Cap at 100

def calculate_clicks(channel, impressions, creative_rating, campaign_type):
    """
    Estimates number of clicks based on impressions and campaign parameters.
    
    Args:
        channel (str): Marketing channel
        impressions (int): Number of impressions
        creative_rating (int): Quality rating of creative assets (1-10)
        campaign_type (str): Type of campaign
    
    Returns:
        int: Estimated number of clicks
    """
    # Base click-through rates by channel and campaign type
    base_ctr = {
        'Social Media': {
            'Brand Awareness': 0.03,
            'Product Launch': 0.04,
            'Promotional': 0.05
        },
        'Email': {
            'Brand Awareness': 0.10,
            'Product Launch': 0.12,
            'Promotional': 0.15
        },
        'TV': 0.0005,  # TV has very low CTR
        'Radio': 0.001,  # Radio has low CTR
        'Search': {
            'Brand Awareness': 0.08,
            'Product Launch': 0.10,
            'Promotional': 0.12
        }
    }
    
    # Creative quality impact on CTR
    creative_factor = 1 + (creative_rating - 5) * 0.10  # 10% change per point from baseline 5
    
    # Get base CTR for this channel and campaign type
    if isinstance(base_ctr.get(channel, 0), dict):
        ctr = base_ctr[channel].get(campaign_type, 0.02)  # Default 2% if combination not found
    else:
        ctr = base_ctr.get(channel, 0.02)  # Default 2% if channel not found
    
    return int(impressions * ctr * creative_factor)  # Return integer clicks

def calculate_revenue(clicks, conversion_rate, product_type):
    """
    Calculates estimated revenue based on clicks, conversion rate, and product type.
    
    Args:
        clicks (int): Number of clicks
        conversion_rate (float): Conversion rate (0-1)
        product_type (str): Type of product being sold
    
    Returns:
        float: Estimated revenue in CLP
    """
    # Average order values by product type (in Chilean Pesos - CLP)
    avg_order_values = {
        'Electronics': 150000,  # ~$150 USD
        'Clothing': 50000,      # ~$50 USD
        'Food': 25000,          # ~$25 USD
        'Services': 100000,     # ~$100 USD
        'Other': 75000          # ~$75 USD
    }
    
    avg_order_value = avg_order_values.get(product_type, 75000)  # Default to 'Other' if not found
    return clicks * conversion_rate * avg_order_value

def calculate_quartile(value, reference_series):
    """
    Calculates which quartile a value falls into based on reference data.
    Handles edge cases gracefully.
    
    Args:
        value (float): Value to evaluate
        reference_series (Series): Reference data for quartile calculation
    
    Returns:
        int: Quartile number (0-3)
    """
    try:
        if len(reference_series.unique()) > 1:
            # Calculate quartiles on reference data and find where value falls
            return pd.qcut(reference_series, 4, labels=False, duplicates='drop')[0]
        return 2  # Default to median if not enough unique values
    except:
        return 2  # Fallback to median on any error

# --------------------------
# Sidebar Configuration
# --------------------------
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="color: #6e8efb; font-size: 22px;">🔧 CONFIGURACIÓN</h2>
            <p style="font-size: 12px; opacity: 0.7;">Desarrollado por Bo Kolstrup</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Debug mode toggle
    debug_mode = st.checkbox("Modo Depuración", 
                           help="Mostrar información de depuración y datos crudos")
    
    # Footer in sidebar
    st.markdown("""
        <div style="margin-top: 30px; text-align: center;">
            <p style="font-size: 12px; opacity: 0.7;">ROI Predictor Pro v1.0</p>
        </div>
    """, unsafe_allow_html=True)

# --------------------------
# Main Input Form
# --------------------------
with st.form("roi_prediction_form"):
    st.subheader("📋 Detalles de la Campaña")
    
    # Split form into two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Campaign attributes
        channel = st.selectbox(
            "Canal Utilizado",
            options=df['Channel_Used'].unique(),
            help="Selecciona el canal de marketing principal para esta campaña"
        )
        
        campaign_type = st.selectbox(
            "Tipo de Campaña",
            options=df['Campaign_Type'].unique(),
            help="El objetivo estratégico de esta campaña"
        )
        
        target_audience = st.selectbox(
            "Audiencia Objetivo",
            options=df['Target_Audience'].unique(),
            help="Grupo demográfico principal al que se dirige"
        )
        
        product_type = st.selectbox(
            "Tipo de Producto",
            options=df['Product_Type'].unique(),
            help="Tipo de producto o servicio que se promociona"
        )
        
        customer_segment = st.selectbox(
            "Segmento de Clientes",
            options=df['Customer_Segment'].unique(),
            help="Segmento de clientes existentes al que se dirige"
        )
    
    with col2:
        # Campaign metrics
        budget = st.number_input(
            "Presupuesto de Campaña (CLP)",
            min_value=1000,
            max_value=100000000,
            value=1000000,
            step=1000,
            help="Presupuesto total asignado para esta campaña en Pesos Chilenos"
        )
        
        duration = st.number_input(
            "Duración de la Campaña (días)",
            min_value=1,
            max_value=365,
            value=30,
            help="Duración de la campaña en días"
        )
        
        impressions = st.number_input(
            "Impresiones Esperadas",
            min_value=100,
            max_value=10000000,
            value=10000,
            step=100,
            help="Número estimado de veces que se verá el anuncio"
        )
        
        creative_rating = st.slider(
            "Calidad Creativa (1-10)",
            min_value=1,
            max_value=10,
            value=7,
            help="Valoración subjetiva de la calidad de los recursos creativos (1=Pobre, 10=Excelente)"
        )
        
        seasonality = st.selectbox(
            "Factor de Estacionalidad",
            options=["Baja", "Media", "Alta", "Máxima"],
            help="Impacto estacional esperado en el rendimiento de la campaña"
        )
    
    # Form submission button
    submitted = st.form_submit_button("🚀 Predecir ROI", use_container_width=True)

# Define all expected columns for the model (including one-hot encoded versions)
expected_columns = [
    # Core metrics
    'Conversion_Rate', 'Clicks', 'Impressions', 'Engagement_Score', 
    'Costo_Adquisicion_CLP', 'Revenue_CLP', 'Calculated_CR', 
    
    # Original and calculated versions
    'Original_Conversion_Rate', 'CR_Discrepancy', 'Adjusted_CTR', 
    
    # Quartile features
    'Clicks_quartile', 'Impressions_quartile', 
    
    # Derived metrics
    'CTR', 'Conv_per_Click', 'Conv_per_Impression', 'Engagement_Index', 
    
    # Transformed features
    'Conversion_Rate_transformed', 'Calculated_CR_transformed', 
    'CR_Discrepancy_transformed', 'CTR_transformed', 'Conv_per_Click_transformed', 
    
    # Original scale features
    'Conversion_Rate_original', 'Revenue_CLP_original', 'Calculated_CR_original', 
    'CR_Discrepancy_original', 'CTR_original', 'Conv_per_Click_original', 
    
    # Time-based features
    'campaign_duration_days', 
    
    # Cost metrics
    'cost_per_click', 'cost_per_impression', 
    
    # Performance metrics
    'click_through_rate', 'revenue_per_click', 'engagement_rate', 
    'cost_engagement_interaction',
    
    # One-hot encoded features (exactly as the model expects)
    'Conversion_Rate_Source_Calculated_CTR',
    'Click_Quality_Normal', 
    'Click_Quality_Suspicious (Round Number)',
    
    # Original categorical columns that were one-hot encoded
    'Conversion_Rate_Source',
    'Click_Quality'
]

# --------------------------
# Prediction and Results Display
# --------------------------
if submitted:
    # Calculate dynamic metrics with enhanced functions
    conversion_rate = calculate_conversion_rate(
        channel, campaign_type, target_audience, product_type, 
        creative_rating, duration, seasonality
    )
    
    engagement_score = calculate_engagement_score(
        channel, duration, budget, seasonality, 
        campaign_type, creative_rating
    )
    
    clicks = calculate_clicks(channel, impressions, creative_rating, campaign_type)
    
    # Calculate revenue independently of ROI prediction
    revenue = calculate_revenue(clicks, conversion_rate, product_type)
    
    # Calculate quartiles using the full dataset as reference
    clicks_quartile = calculate_quartile(clicks, df['Clicks'])
    impressions_quartile = calculate_quartile(impressions, df['Impressions'])
    
    # Create complete input dataframe with all expected features
    input_data = pd.DataFrame([{
        # Core features
        'Channel_Used': channel,
        'Campaign_Type': campaign_type,
        'Target_Audience': target_audience,
        'Company': df['Company'].mode()[0],  # Use most common company
        'Location': df['Location'].mode()[0],  # Use most common location
        'Customer_Segment': customer_segment,
        'Product_Type': product_type,
        'Duration': f"{duration} days",
        'start_date': pd.Timestamp.now(),
        'end_date': pd.Timestamp.now() + pd.Timedelta(days=duration),
        'CR_Outlier': False,
        'Creative_Score': creative_rating,
        'Seasonality_Factor': seasonality,
        
        # Main metrics
        'Costo_Adquisicion_CLP': budget,
        'Impressions': impressions,
        'campaign_duration_days': duration,
        'Conversion_Rate': conversion_rate,
        'Revenue_CLP': revenue,
        'Engagement_Score': engagement_score,
        'Clicks': clicks,
        
        # Original and calculated metrics
        'Original_Conversion_Rate': conversion_rate,
        'Calculated_CR': conversion_rate,
        'CR_Discrepancy': 0,
        'Adjusted_CTR': clicks/impressions if impressions > 0 else 0,
        
        # Quartiles
        'Clicks_quartile': clicks_quartile,
        'Impressions_quartile': impressions_quartile,
        
        # Derived metrics
        'CTR': clicks/impressions if impressions > 0 else 0,
        'Conv_per_Click': conversion_rate / clicks if clicks > 0 else 0,
        'Conv_per_Impression': conversion_rate / impressions if impressions > 0 else 0,
        'Engagement_Index': engagement_score,
        'cost_per_click': budget / clicks if clicks > 0 else 0,
        'cost_per_impression': budget / impressions if impressions > 0 else 0,
        'click_through_rate': clicks/impressions if impressions > 0 else 0,
        'revenue_per_click': revenue / clicks if clicks > 0 else 0,
        'engagement_rate': engagement_score / 100,
        'cost_engagement_interaction': budget * engagement_score,
        
        # Transformed features
        'Conversion_Rate_transformed': np.log1p(conversion_rate),
        'Calculated_CR_transformed': np.log1p(conversion_rate),
        'CR_Discrepancy_transformed': 0,
        'CTR_transformed': np.log1p(clicks/impressions) if impressions > 0 and clicks > 0 else 0,
        'Conv_per_Click_transformed': np.log1p(conversion_rate / clicks) if clicks > 0 and conversion_rate > 0 else 0,
        
        # Original versions
        'Conversion_Rate_original': conversion_rate,
        'Revenue_CLP_original': revenue,
        'Calculated_CR_original': conversion_rate,
        'CR_Discrepancy_original': 0,
        'CTR_original': clicks/impressions if impressions > 0 else 0,
        'Conv_per_Click_original': conversion_rate / clicks if clicks > 0 else 0,
        
        # One-hot encoded features (exactly as model expects)
        'Conversion_Rate_Source_Calculated_CTR': 1,
        'Click_Quality_Normal': 1,
        'Click_Quality_Suspicious (Round Number)': 0,
        
        # Original categorical columns that were one-hot encoded
        'Conversion_Rate_Source': 'Calculated_CTR',
        'Click_Quality': 'Normal'
    }])
    
    # Add any missing columns with default values
    for col in expected_columns:
        if col not in input_data.columns:
            if 'transformed' in col:
                input_data[col] = 0  # Default for transformed features
            elif 'original' in col:
                # Copy from non-original version
                base_col = col.replace('_original', '')
                input_data[col] = input_data[base_col] if base_col in input_data.columns else 0
            elif col == 'Click_Quality_Normal':
                input_data[col] = 1
            elif col == 'Click_Quality_Suspicious (Round Number)':
                input_data[col] = 0
            elif col == 'Conversion_Rate_Source_Calculated_CTR':
                input_data[col] = 1
            else:
                input_data[col] = 0
    
    # Ensure correct data types - only convert numeric columns
    for col in input_data.columns:
        if col in expected_columns:
            if col in ['Conversion_Rate_Source', 'Click_Quality']:
                # Keep categorical columns as strings
                continue
            elif col in ['Conversion_Rate_Source_Calculated_CTR', 'Click_Quality_Normal', 
                        'Click_Quality_Suspicious (Round Number)']:
                # Ensure one-hot encoded columns remain as integers
                input_data[col] = input_data[col].astype(int)
            else:
                # Convert all other expected columns to float
                input_data[col] = input_data[col].astype(float)
    
    # Reorder columns to match model expectations
    input_data = input_data[expected_columns]
    
    # Store input data in session state for scenario analysis
    st.session_state.input_data = input_data.copy()
    
    # Debug information if debug mode is enabled
    if debug_mode:
        st.subheader("🔍 Información de Depuración")
        with st.expander("Vista Previa de Datos de Entrada"):
            st.dataframe(input_data)
            
        with st.expander("Validación de Datos"):
            # Check for NaN values
            st.write("Comprobación de Valores Faltantes:")
            st.write(input_data.isnull().sum())
            
            # Check data types
            st.write("Tipos de Datos:")
            st.write(input_data.dtypes)
            
            # Check column match
            st.write("Columnas en entrada vs esperadas:")
            st.write(pd.DataFrame({
                'En Entrada': [col in input_data.columns for col in expected_columns],
                'Esperadas': expected_columns
            }))
    
    try:
        # Make prediction using the loaded model
        st.session_state.roi_prediction = model.predict(input_data)[0]
        
        # Display prediction in styled cards
        st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("ROI Predicho", 
                      f"{st.session_state.roi_prediction:.2f}x", 
                      help_text="Predicción de Retorno de la Inversión")
        with col2:
            metric_card("Ingresos Estimados", 
                      f"CLP {revenue:,.0f}", 
                      help_text="Ingresos proyectados de la campaña")
        with col3:
            metric_card("Tasa de Conversión", 
                      f"{conversion_rate:.2%}", 
                      help_text="Tasa de conversión estimada")
        with col4:
            metric_card("Clics Estimados", 
                      f"{clicks:,.0f}", 
                      help_text="Clics proyectados de la campaña")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Apply metric card styling if extras are available
        if EXTRAS_AVAILABLE:
            extras['style_metric_cards'](border_left_color="#6e8efb", background_color="#121212")
        
    except Exception as e:
        st.error(f"Error al realizar la predicción: {str(e)}")
        if debug_mode:
            st.exception(e)  # Show full traceback in debug mode

# --------------------------
# Scenario Analysis Section
# --------------------------
st.subheader("📊 Análisis de Escenarios")
st.markdown("""
Ajusta los parámetros de la campaña para ver cómo podrían afectar el ROI.
""")

# Scenario adjustment sliders
col1, col2, col3 = st.columns(3)

with col1:
    budget_change = st.slider(
        "Cambio de Presupuesto (%)",
        min_value=-50,  # Can reduce budget by up to 50%
        max_value=200,   # Or increase by up to 200%
        value=0,         # Default to no change
        step=5           # 5% increments
    )

with col2:
    duration_change = st.slider(
        "Cambio de Duración (%)",
        min_value=-50,
        max_value=200,
        value=0,
        step=5
    )

with col3:
    creative_change = st.slider(
        "Cambio en Calidad Creativa (puntos)",
        min_value=-3,  # Can decrease by up to 3 points
        max_value=3,    # Or increase by up to 3 points
        value=0,        # Default to no change
        step=1          # 1-point increments
    )

# Scenario analysis button
if st.button("🔍 Analizar Escenario", use_container_width=True):
    if st.session_state.roi_prediction is not None and st.session_state.input_data is not None:
        original_input = st.session_state.input_data.copy()
        original_budget = original_input['Costo_Adquisicion_CLP'].values[0]
        original_duration = original_input['campaign_duration_days'].values[0]
        original_impressions = original_input['Impressions'].values[0]
        original_creative_rating = original_input['Creative_Score'].values[0] if 'Creative_Score' in original_input.columns else 7
        original_seasonality = original_input['Seasonality_Factor'].values[0] if 'Seasonality_Factor' in original_input.columns else "Medium"
        original_product_type = original_input['Product_Type'].values[0] if 'Product_Type' in original_input.columns else "Other"
        
        # Calculate adjusted parameters based on slider changes
        adjusted_budget = original_budget * (1 + budget_change/100)
        adjusted_duration = original_duration * (1 + duration_change/100)
        adjusted_creative_rating = max(min(original_creative_rating + creative_change, 10), 1)  # Keep within 1-10 range
        adjusted_impressions = original_impressions * (1 + budget_change/100 * 0.7)  # 70% correlation between budget and impressions
        
        # Recalculate metrics with new parameters
        adjusted_conversion_rate = calculate_conversion_rate(
            channel,
            campaign_type,
            target_audience,
            original_product_type,
            adjusted_creative_rating,
            adjusted_duration,
            original_seasonality
        )
        
        adjusted_engagement_score = calculate_engagement_score(
            channel,
            adjusted_duration,
            adjusted_budget,
            original_seasonality,
            campaign_type,
            adjusted_creative_rating
        )
        
        adjusted_clicks = calculate_clicks(
            channel,
            adjusted_impressions,
            adjusted_creative_rating,
            campaign_type
        )
        
        adjusted_revenue = calculate_revenue(
            adjusted_clicks,
            adjusted_conversion_rate,
            original_product_type
        )
        
        # Calculate new quartiles
        adjusted_clicks_quartile = calculate_quartile(adjusted_clicks, df['Clicks'])
        adjusted_impressions_quartile = calculate_quartile(adjusted_impressions, df['Impressions'])
        
        # Create adjusted input dataframe
        adjusted_input = original_input.copy()
        
        # Update all features that depend on the changed parameters
        adjusted_input['Costo_Adquisicion_CLP'] = adjusted_budget
        adjusted_input['campaign_duration_days'] = adjusted_duration
        adjusted_input['Impressions'] = adjusted_impressions
        adjusted_input['Clicks'] = adjusted_clicks
        adjusted_input['Conversion_Rate'] = adjusted_conversion_rate
        adjusted_input['Engagement_Score'] = adjusted_engagement_score
        adjusted_input['Revenue_CLP'] = adjusted_revenue
        adjusted_input['Clicks_quartile'] = adjusted_clicks_quartile
        adjusted_input['Impressions_quartile'] = adjusted_impressions_quartile
        adjusted_input['Creative_Score'] = adjusted_creative_rating
        
        # Update all derived metrics
        adjusted_input['Adjusted_CTR'] = adjusted_clicks/adjusted_impressions if adjusted_impressions > 0 else 0
        adjusted_input['CTR'] = adjusted_clicks/adjusted_impressions if adjusted_impressions > 0 else 0
        adjusted_input['Conv_per_Click'] = adjusted_conversion_rate / adjusted_clicks if adjusted_clicks > 0 else 0
        adjusted_input['Conv_per_Impression'] = adjusted_conversion_rate / adjusted_impressions if adjusted_impressions > 0 else 0
        adjusted_input['Engagement_Index'] = adjusted_engagement_score
        adjusted_input['cost_per_click'] = adjusted_budget / adjusted_clicks if adjusted_clicks > 0 else 0
        adjusted_input['cost_per_impression'] = adjusted_budget / adjusted_impressions if adjusted_impressions > 0 else 0
        adjusted_input['click_through_rate'] = adjusted_clicks/adjusted_impressions if adjusted_impressions > 0 else 0
        adjusted_input['revenue_per_click'] = adjusted_revenue / adjusted_clicks if adjusted_clicks > 0 else 0
        adjusted_input['engagement_rate'] = adjusted_engagement_score / 100
        adjusted_input['cost_engagement_interaction'] = adjusted_budget * adjusted_engagement_score
        
        # Update transformed features
        adjusted_input['Conversion_Rate_transformed'] = np.log1p(adjusted_conversion_rate)
        adjusted_input['Calculated_CR_transformed'] = np.log1p(adjusted_conversion_rate)
        adjusted_input['CTR_transformed'] = np.log1p(adjusted_clicks/adjusted_impressions) if adjusted_impressions > 0 and adjusted_clicks > 0 else 0
        adjusted_input['Conv_per_Click_transformed'] = np.log1p(adjusted_conversion_rate / adjusted_clicks) if adjusted_clicks > 0 and adjusted_conversion_rate > 0 else 0
        
        # Update original versions
        adjusted_input['Conversion_Rate_original'] = adjusted_conversion_rate
        adjusted_input['Revenue_CLP_original'] = adjusted_revenue
        adjusted_input['Calculated_CR_original'] = adjusted_conversion_rate
        adjusted_input['CTR_original'] = adjusted_clicks/adjusted_impressions if adjusted_impressions > 0 else 0
        adjusted_input['Conv_per_Click_original'] = adjusted_conversion_rate / adjusted_clicks if adjusted_clicks > 0 else 0
        
        # Ensure all expected columns are present
        for col in expected_columns:
            if col not in adjusted_input.columns:
                if 'transformed' in col:
                    adjusted_input[col] = 0
                elif 'original' in col:
                    base_col = col.replace('_original', '')
                    adjusted_input[col] = adjusted_input[base_col] if base_col in adjusted_input.columns else 0
                elif col == 'Click_Quality_Normal':
                    adjusted_input[col] = 1
                elif col == 'Click_Quality_Suspicious (Round Number)':
                    adjusted_input[col] = 0
                elif col == 'Conversion_Rate_Source_Calculated_CTR':
                    adjusted_input[col] = 1
                else:
                    adjusted_input[col] = 0
        
        # Reorder columns to match model expectations
        adjusted_input = adjusted_input[expected_columns]
        
        # Make adjusted prediction
        adjusted_roi = model.predict(adjusted_input)[0]
        
        # Display comparison in styled cards
        st.markdown("### Comparación de Escenarios")
        
        # ROI and Budget comparison
        st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("ROI Original", 
                       f"{st.session_state.roi_prediction:.2f}x",
                       help_text="Predicción original")
        with col2:
            metric_card("ROI Ajustado", 
                       f"{adjusted_roi:.2f}x", 
                       delta=f"{(adjusted_roi - st.session_state.roi_prediction):.2f}x",
                       help_text="Predicción del escenario")
        with col3:
            metric_card("Presupuesto Original", 
                       f"CLP {original_budget:,.0f}",
                       help_text="Presupuesto inicial de la campaña")
        with col4:
            metric_card("Presupuesto Ajustado", 
                       f"CLP {adjusted_budget:,.0f}", 
                       delta=f"{budget_change}%",
                       help_text="Presupuesto del escenario")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Duration and Creative comparison
        st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            metric_card("Duración Original", 
                       f"{original_duration} días",
                       help_text="Duración inicial de la campaña")
        with col6:
            metric_card("Duración Ajustada", 
                       f"{adjusted_duration:.0f} días", 
                       delta=f"{duration_change}%",
                       help_text="Duración del escenario")
        with col7:
            metric_card("Creatividad Original", 
                       f"{original_creative_rating}/10",
                       help_text="Valoración creativa inicial")
        with col8:
            metric_card("Creatividad Ajustada", 
                       f"{adjusted_creative_rating}/10", 
                       delta=f"{creative_change} puntos",
                       help_text="Valoración creativa del escenario")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Apply styling if extras are available
        if EXTRAS_AVAILABLE:
            extras['style_metric_cards'](border_left_color="#6e8efb", background_color="#121212")
        
        # Performance metrics comparison table
        st.markdown("### Impacto en el Rendimiento")
        metrics = pd.DataFrame({
            'Métrica': ['Impresiones', 'Clics', 'CTR', 'Tasa de Conversión', 'Puntuación de Engagement', 'Ingresos'],
            'Original': [
                f"{original_impressions:,.0f}",
                f"{original_input['Clicks'].values[0]:,.0f}",
                f"{original_input['CTR'].values[0]:.2%}",
                f"{original_input['Conversion_Rate'].values[0]:.2%}",
                f"{original_input['Engagement_Score'].values[0]:.0f}",
                f"CLP {original_input['Revenue_CLP'].values[0]:,.0f}"
            ],
            'Ajustado': [
                f"{adjusted_impressions:,.0f}",
                f"{adjusted_clicks:,.0f}",
                f"{adjusted_clicks/adjusted_impressions:.2%}" if adjusted_impressions > 0 else "0%",
                f"{adjusted_conversion_rate:.2%}",
                f"{adjusted_engagement_score:.0f}",
                f"CLP {adjusted_revenue:,.0f}"
            ],
            'Cambio': [
                f"{(adjusted_impressions - original_impressions)/original_impressions:.1%}" if original_impressions > 0 else "N/A",
                f"{(adjusted_clicks - original_input['Clicks'].values[0])/original_input['Clicks'].values[0]:.1%}" if original_input['Clicks'].values[0] > 0 else "N/A",
                f"{(adjusted_clicks/adjusted_impressions - original_input['CTR'].values[0])/original_input['CTR'].values[0]:.1%}" if original_input['CTR'].values[0] > 0 else "N/A",
                f"{(adjusted_conversion_rate - original_input['Conversion_Rate'].values[0])/original_input['Conversion_Rate'].values[0]:.1%}",
                f"{(adjusted_engagement_score - original_input['Engagement_Score'].values[0])/original_input['Engagement_Score'].values[0]:.1%}",
                f"{(adjusted_revenue - original_input['Revenue_CLP'].values[0])/original_input['Revenue_CLP'].values[0]:.1%}"
            ]
        })
        
        # Display metrics comparison table
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        
        # Show debug info if debug mode is enabled
        if debug_mode:
            st.subheader("🔍 Información de Depuración del Escenario")
            with st.expander("Datos de Entrada Ajustados"):
                st.dataframe(adjusted_input)
            
    else:
        st.warning("Por favor, realiza una predicción inicial antes de analizar escenarios.")

# --------------------------
# Footer with Developer Info
# --------------------------
st.markdown("---")
st.markdown("""
    <div style="text-align: center; font-size: 0.9em; opacity: 0.8;">
    <p>Desarrollado por Bo Kolstrup | Científico de Datos</p>
    <p>Contacto: bokolstrup@gmail.com | +56 9 4259 6282</p>
    <p>Última actualización: {date}</p>
    </div>
""".format(date=datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)