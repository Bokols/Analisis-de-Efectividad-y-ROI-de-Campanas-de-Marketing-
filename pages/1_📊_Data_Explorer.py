#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Panel de Control Marketing ROI Explorer Pro

Un panel analítico completo para evaluar el rendimiento de campañas de marketing
en múltiples dimensiones incluyendo ROI, métricas de conversión, engagement y tendencias.

Características:
- Sistema de filtrado interactivo
- Tarjetas de KPI con tooltips fijos
- Sistema de visualización con pestañas múltiples
- Capacidades de exploración de datos
- Carga y procesamiento optimizado de datos
- Diseño responsivo
- Manejo robusto de datos vacíos

Autor: Bo Kolstrup
Versión: 1.3.2
Última actualización: 2024-06-30
"""

# --------------------------
# Core Imports
# --------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from packaging import version
import warnings

# Suppress FutureWarnings from pandas
warnings.filterwarnings('ignore', category=FutureWarning)

# --------------------------
# Tooltip Compatibility Layer
# --------------------------
def add_tooltip(text, help_text):
    """Añade un tooltip usando las características disponibles de Streamlit según la versión."""
    try:
        # Try using native tooltip if available (Streamlit >= 1.25.0)
        if hasattr(st, 'tooltip'):
            with st.tooltip(help_text):
                st.markdown(f"{text} ⓘ", unsafe_allow_html=True)
        else:
            # Fallback to using markdown with hover text
            st.markdown(f'<span title="{help_text}">{text} ⓘ</span>', unsafe_allow_html=True)
    except:
        # Final fallback - just show the text
        st.markdown(text, unsafe_allow_html=True)

# --------------------------
# Initial Configuration
# --------------------------
st.set_page_config(
    page_title="🚀 Explorador de Campañas Marketing Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Optimized Imports with Lazy Loading
# --------------------------
def load_extras():
    """Carga dinámica de módulos extras de Streamlit con manejo de errores."""
    try:
        from streamlit_extras.metric_cards import style_metric_cards
        from streamlit_extras.dataframe_explorer import dataframe_explorer
        from streamlit_extras.stylable_container import stylable_container
        import streamlit_extras
        
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
        return {
            'extras_available': False,
            'requires_key': False
        }

extras = load_extras()
EXTRAS_AVAILABLE = extras['extras_available']
REQUIRES_KEY = extras.get('requires_key', False)

# --------------------------
# Optimized Data Loading with Caching
# --------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Carga y preprocesa datos de campañas de marketing."""
    try:
        usecols = [
            'start_date', 'end_date', 'Costo_Adquisicion_CLP', 'Clicks',
            'Impressions', 'ROI', 'Channel_Used', 'Campaign_Type',
            'Conversion_Rate', 'Engagement_Score'
        ]
        
        # Updated to use Parquet format
        df = pd.read_parquet(
            "data/marketing_data_for_powerbi.parquet",
            columns=usecols
        ).dropna(subset=['start_date', 'end_date'])
        
        # Ensure dates are properly formatted
        df['start_date'] = pd.to_datetime(df['start_date'], format='%Y-%m-%d', errors='coerce')
        df['end_date'] = pd.to_datetime(df['end_date'], format='%Y-%m-%d', errors='coerce')
        
        df['campaign_duration_days'] = (df['end_date'] - df['start_date']).dt.days
        df['CPC'] = np.where(df['Clicks'] > 0, 
                           df['Costo_Adquisicion_CLP'] / df['Clicks'], 
                           np.nan)
        df['CTR'] = np.where(df['Impressions'] > 0,
                           (df['Clicks'] / df['Impressions']) * 100,
                           np.nan)
        
        roi_bins = [-np.inf, 1, 3, 5, np.inf]
        roi_labels = ['Pobre (<1x)', 'Moderado (1-3x)', 'Bueno (3-5x)', 'Excelente (>5x)']
        df['ROI_category'] = pd.cut(df['ROI'], bins=roi_bins, labels=roi_labels)
        
        return df, roi_labels
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return pd.DataFrame(), []

# --------------------------
# UI Component Builders
# --------------------------
def get_stylable_container(styles, container_key=None):
    """Crea un contenedor estilizado con fallback."""
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

# --------------------------
# Data Loading
# --------------------------
with st.spinner('Cargando datos...'):
    df, roi_labels = load_data()

# --------------------------
# Custom CSS for Styling
# --------------------------
st.markdown("""
    <style>
        .main { background-color: #000000; color: #ffffff; }
        .sidebar .sidebar-content { 
            background-color: #121212; 
            border-right: 1px solid #333333;
        }
        .stMetric { 
            background-color: #121212 !important; 
            border-radius: 10px;
        }
        .stDataFrame { 
            background-color: #121212;
            border: 1px solid #333333;
        }
        .stTabs [data-baseweb="tab"] { 
            background-color: #121212 !important; 
            border-bottom: 1px solid #333333;
        }
        .stTabs [aria-selected="true"] { 
            background-color: #1e1e1e !important; 
            border-bottom: 2px solid #6e8efb;
        }
        .stSlider > div > div {
            background-color: #6e8efb;
        }
        .stSelectbox, .stMultiselect {
            background-color: #121212;
        }
        
        /* KPI Grid Container */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        /* KPI Card Styling */
        .kpi-card {
            background-color: #121212;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #333333;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            transition: transform 0.2s, box-shadow 0.2s;
            position: relative;
            overflow: visible;
        }
        
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.4);
            border-color: #6e8efb;
        }
        
        .kpi-title {
            font-size: 1rem;
            margin-bottom: 8px;
            color: #6e8efb;
            display: flex;
            align-items: center;
        }
        
        .kpi-value {
            font-size: 1.8rem;
            font-weight: bold;
            margin: 5px 0;
            color: white;
        }
        
        .kpi-delta {
            font-size: 0.9rem;
            color: #aaa;
        }
        
        /* Tooltip Styling */
        .kpi-tooltip {
            position: relative;
            display: inline-block;
            margin-left: 4px;
        }
        
        .kpi-tooltip .tooltip-icon {
            color: #6e8efb;
            cursor: help;
            font-size: 0.8em;
            vertical-align: middle;
        }
        
        .kpi-tooltip .tooltip-text {
            visibility: hidden;
            width: 200px;
            background-color: #1e1e1e;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            border: 1px solid #333333;
            font-size: 0.9rem;
            font-weight: normal;
            pointer-events: none;
        }
        
        .kpi-tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
        
        /* Responsive adjustments */
        @media (max-width: 1200px) {
            .kpi-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        
        @media (max-width: 800px) {
            .kpi-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .kpi-value {
                font-size: 1.5rem;
            }
        }
        
        @media (max-width: 500px) {
            .kpi-grid {
                grid-template-columns: 1fr;
            }
        }
        
        /* Warning box styling */
        .data-warning {
            background-color: #332222;
            border-left: 4px solid #ff4b4b;
            padding: 16px;
            margin: 20px 0;
            border-radius: 4px;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Header Section
# --------------------------
col1, col2 = st.columns([5,1])
with col1:
    st.title("🚀 Explorador de Campañas Marketing Pro")
    add_tooltip("""
        <div style="opacity: 0.9; font-size: 16px;">
        Panel analítico avanzado para optimizar el rendimiento de marketing en diferentes canales,
        industrias y periodos de tiempo. Descubre insights para maximizar tu ROI.
        </div>
    """, "Analiza el rendimiento de marketing por canales y periodos de tiempo")
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/3281/3281289.png", width=80)

# --------------------------
# Sidebar Filters
# --------------------------
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="color: #6e8efb; font-size: 22px;">🔍 FILTROS</h2>
            <p style="font-size: 12px; opacity: 0.7;">Desarrollado por Bo Kolstrup</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Date Range Filter
    with st.expander("📅 Rango de Fechas", expanded=True):
        add_tooltip("Rango de Fechas", "Selecciona el rango de fechas para el análisis")
        date_range = st.date_input(
            "Seleccionar Rango de Fechas",
            value=[df['start_date'].min(), df['start_date'].max()],
            min_value=df['start_date'].min(),
            max_value=df['start_date'].max(),
            label_visibility="collapsed"
        )
    
    # Campaign Attribute Filters
    with st.expander("📊 Atributos de Campaña", expanded=True):
        add_tooltip("Canales", "Filtrar por canales de marketing")
        selected_channels = st.multiselect(
            "Seleccionar Canales",
            options=df['Channel_Used'].unique(),
            default=df['Channel_Used'].unique(),
            key="channels"
        )
        
        add_tooltip("Tipos de Campaña", "Filtrar por tipos de campaña")
        selected_campaign_types = st.multiselect(
            "Seleccionar Tipos de Campaña",
            options=df['Campaign_Type'].unique(),
            default=df['Campaign_Type'].unique(),
            key="campaign_types"
        )
        
        add_tooltip("Categorías ROI", "Filtrar por categorías de rendimiento de ROI")
        roi_categories = st.multiselect(
            "Categorías ROI",
            options=roi_labels,
            default=roi_labels,
            key="roi_cats"
        )
    
    # Performance Range Filters
    with st.expander("📈 Rango de Rendimiento", expanded=True):
        add_tooltip("Rango ROI", "Filtrar por rango de ROI (retorno de inversión)")
        roi_range = st.slider(
            "Rango ROI",
            min_value=float(df['ROI'].min()),
            max_value=float(df['ROI'].max()),
            value=(float(df['ROI'].min()), float(df['ROI'].max())),
            step=0.1,
            key="roi_slider"
        )
        
        add_tooltip("Rango de Gasto", "Filtrar por rango de gasto en campañas en CLP (Pesos Chilenos)")
        spend_range = st.slider(
            "Rango de Gasto (CLP)",
            min_value=float(df['Costo_Adquisicion_CLP'].min()),
            max_value=float(df['Costo_Adquisicion_CLP'].max()),
            value=(float(df['Costo_Adquisicion_CLP'].min()), float(df['Costo_Adquisicion_CLP'].max())),
            step=1000.0,
            format="%.0f"
        )

# --------------------------
# Data Filtering Function
# --------------------------
@st.cache_data
def filter_data(df, date_range, selected_channels, selected_campaign_types, 
               roi_categories, roi_range, spend_range):
    """Aplica todos los filtros seleccionados por el usuario al dataset."""
    # Handle empty filters by defaulting to all options
    selected_channels = selected_channels or df['Channel_Used'].unique()
    selected_campaign_types = selected_campaign_types or df['Campaign_Type'].unique()
    roi_categories = roi_categories or roi_labels
    
    return df[
        (df['Channel_Used'].isin(selected_channels)) &
        (df['Campaign_Type'].isin(selected_campaign_types)) &
        (df['start_date'] >= pd.to_datetime(date_range[0])) &
        (df['start_date'] <= pd.to_datetime(date_range[1])) &
        (df['ROI'] >= roi_range[0]) &
        (df['ROI'] <= roi_range[1]) &
        (df['Costo_Adquisicion_CLP'] >= spend_range[0]) &
        (df['Costo_Adquisicion_CLP'] <= spend_range[1]) &
        (df['ROI_category'].isin(roi_categories))
    ]

filtered_df = filter_data(df, date_range, selected_channels, selected_campaign_types,
                         roi_categories, roi_range, spend_range)

# --------------------------
# Handle empty dataset scenario
# --------------------------
if filtered_df.empty:
    st.markdown("""
        <div class="data-warning">
            <h3>⚠️ Sin datos disponibles</h3>
            <p>Los filtros seleccionados no devuelven ningún dato. Por favor:</p>
            <ul>
                <li>Amplíe el rango de fechas</li>
                <li>Seleccione más canales o tipos de campaña</li>
                <li>Relaje los rangos de ROI o gasto</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Skip rest of the app
    st.stop()

# --------------------------
# Improved KPI Cards Section
# --------------------------
def improved_metric_card(title, value, help_text=None, delta=None):
    """Tarjeta de métricas rediseñada con tooltips confiables y mejor estilo."""
    tooltip_html = ""
    if help_text:
        tooltip_html = f"""
        <span class="kpi-tooltip">
            <span class="tooltip-icon">ⓘ</span>
            <span class="tooltip-text">{help_text}</span>
        </span>
        """
    
    delta_html = f"<div class='kpi-delta'>{delta}</div>" if delta else ""
    
    card_html = f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}{tooltip_html}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# KPI Cards Section
st.markdown("## 📊 Resumen de Rendimiento")

# First Row - 3 KPI Cards
col1, col2, col3 = st.columns(3)
with col1:
    improved_metric_card(
        "Total Campañas", 
        f"{len(filtered_df):,}", 
        "Número de campañas que coinciden con los filtros actuales"
    )
with col2:
    improved_metric_card(
        "Gasto Total", 
        f"CLP {filtered_df['Costo_Adquisicion_CLP'].sum()/1e6:,.1f}M", 
        "Costo total de adquisición en las campañas seleccionadas"
    )
with col3:
    delta = (filtered_df['ROI'].mean() - df['ROI'].mean()) / df['ROI'].mean() * 100
    improved_metric_card(
        "ROI Promedio", 
        f"{filtered_df['ROI'].mean():.1f}x", 
        "Retorno de inversión (ingresos/costo)", 
        f"{delta:.1f}% vs general"
    )

# Second Row - 3 KPI Cards
col4, col5, col6 = st.columns(3)
with col4:
    improved_metric_card(
        "Clics Totales", 
        f"{filtered_df['Clicks'].sum():,}", 
        "Clics totales generados por las campañas seleccionadas"
    )
with col5:
    improved_metric_card(
        "Impresiones Totales", 
        f"{filtered_df['Impressions'].sum():,}", 
        "Impresiones totales de las campañas seleccionadas"
    )
with col6:
    improved_metric_card(
        "CTR Promedio", 
        f"{filtered_df['CTR'].mean():.1f}%", 
        "Tasa de clics promedio (clics/impresiones)"
    )

# Third Row - 4 KPI Cards
col7, col8, col9, col10 = st.columns(4)
with col7:
    improved_metric_card(
        "CPC Promedio", 
        f"CLP {filtered_df['CPC'].mean():,.0f}", 
        "Costo promedio por clic"
    )
with col8:
    improved_metric_card(
        "Conversión Promedio", 
        f"{filtered_df['Conversion_Rate'].mean():.1%}", 
        "Tasa de conversión promedio"
    )
with col9:
    improved_metric_card(
        "Engagement Promedio", 
        f"{filtered_df['Engagement_Score'].mean():.1f}", 
        "Puntuación promedio de engagement (escala 0-10)"
    )
with col10:
    improved_metric_card(
        "Duración Promedio", 
        f"{filtered_df['campaign_duration_days'].mean():.1f} días", 
        "Duración promedio de campañas en días"
    )
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------
# Tab System Implementation
# --------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Análisis ROI", 
    "🔄 Métricas Conversión", 
    "💡 Engagement", 
    "📅 Tendencias Temporales"
])

with tab1:
    col1, col2 = st.columns([6,4])
    with col1:
        add_tooltip("ROI por Canal", "Compara la distribución del rendimiento de ROI entre diferentes canales")
        if not filtered_df['Channel_Used'].empty:
            fig = px.box(filtered_df, x='Channel_Used', y='ROI', 
                        title="<b>Distribución ROI por Canal</b>",
                        color='Channel_Used',
                        color_discrete_sequence=px.colors.qualitative.Dark24,
                        template="plotly_dark")
            fig.update_layout(height=450, font=dict(color='white'), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos disponibles para mostrar ROI por canal")
    
    with col2:
        add_tooltip("Distribución ROI", "Distribución de valores de ROI categorizados por nivel de rendimiento")
        if not filtered_df.empty and 'ROI_category' in filtered_df:
            fig = px.histogram(filtered_df, x='ROI', 
                              title="<b>Distribución ROI</b>",
                              nbins=30,
                              color='ROI_category',
                              color_discrete_sequence=px.colors.sequential.Plasma,
                              template="plotly_dark")
            fig.update_layout(height=450, font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos disponibles para mostrar la distribución de ROI")
    
    add_tooltip("ROI vs Gasto", "Explora la relación entre gasto en campañas y ROI (tamaño de burbuja = impresiones)")
    if len(filtered_df) > 0:
        plot_df = filtered_df.sample(min(1000, len(filtered_df))) if len(filtered_df) > 1000 else filtered_df
        fig = px.scatter(plot_df, 
                        x='Costo_Adquisicion_CLP', 
                        y='ROI', 
                        size='Impressions',
                        color='Channel_Used',
                        log_x=True,
                        title="<b>ROI vs Gasto en Campañas</b>",
                        hover_data=['Campaign_Type', 'Conversion_Rate'],
                        template="plotly_dark")
        fig.update_layout(height=500, font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos suficientes para mostrar ROI vs Gasto")

with tab2:
    @st.cache_data
    def compute_conversion_metrics(df):
        return {
            'conversion': df.groupby('Channel_Used')['Conversion_Rate'].mean().reset_index(),
            'ctr': df.groupby('Channel_Used')['CTR'].mean().reset_index()
        }
    
    metrics = compute_conversion_metrics(filtered_df)
    
    col1, col2 = st.columns(2)
    with col1:
        add_tooltip("Tasas de Conversión", "Compara tasas de conversión entre diferentes canales de marketing")
        if not metrics['conversion'].empty:
            fig = px.bar(metrics['conversion'],
                        x='Channel_Used', 
                        y='Conversion_Rate',
                        title="<b>Tasa de Conversión Promedio por Canal</b>",
                        color='Channel_Used',
                        template="plotly_dark")
            fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
            fig.update_layout(height=400, showlegend=False, font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos disponibles para mostrar la tasa de conversión")
    
    with col2:
        add_tooltip("CTR por Canal", "Compara tasas de clics entre diferentes canales de marketing")
        if not metrics['ctr'].empty:
            fig = px.bar(metrics['ctr'],
                        x='Channel_Used', 
                        y='CTR',
                        title="<b>CTR Promedio por Canal</b>",
                        color='Channel_Used',
                        template="plotly_dark")
            fig.update_traces(texttemplate='%{y:.1f}', textposition='outside')
            fig.update_layout(height=400, showlegend=False, font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos disponibles para mostrar CTR por canal")
    
    add_tooltip("Clics vs Impresiones", "Analiza la relación entre impresiones y clics")
    if len(filtered_df) > 0:
        plot_df = filtered_df.sample(min(1000, len(filtered_df))) if len(filtered_df) > 1000 else filtered_df
        fig = px.scatter(plot_df, 
                        x='Impressions', 
                        y='Clicks', 
                        color='Channel_Used',
                        title="<b>Clics vs Impresiones</b>",
                        hover_data=['Campaign_Type', 'CTR'],
                        template="plotly_dark")
        fig.update_layout(height=500, font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos suficientes para comparar clics e impresiones")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        add_tooltip("Engagement por Canal", "Compara distribuciones de puntuación de engagement entre canales")
        if not filtered_df.empty:
            fig = px.box(filtered_df, x='Channel_Used', y='Engagement_Score',
                        title="<b>Puntuación de Engagement por Canal</b>",
                        color='Channel_Used',
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        template="plotly_dark")
            fig.update_layout(height=450, font=dict(color='white'), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos disponibles para mostrar engagement por canal")
    
    with col2:
        add_tooltip("Engagement vs Clics", "Explora la relación entre clics y puntuaciones de engagement")
        if len(filtered_df) > 0:
            fig = px.scatter(filtered_df, x='Clicks', y='Engagement_Score',
                            color='Channel_Used',
                            title="<b>Engagement vs Clics</b>",
                            hover_data=['Campaign_Type', 'Conversion_Rate'],
                            template="plotly_dark")
            fig.update_layout(height=450, font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para comparar engagement y clics")
    
    add_tooltip("Distribución de Engagement", "Distribución de puntuaciones de engagement coloreadas por canal")
    if not filtered_df.empty:
        fig = px.histogram(filtered_df, x='Engagement_Score',
                          nbins=20,
                          color='Channel_Used',
                          title="<b>Distribución de Puntuaciones de Engagement</b>",
                          template="plotly_dark")
        fig.update_layout(height=450, font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos disponibles para mostrar la distribución de engagement")

with tab4:
    if not filtered_df.empty:
        trends_df = filtered_df.set_index('start_date')
        
        monthly_trends = trends_df.resample('M').agg({
            'ROI': 'mean',
            'Costo_Adquisicion_CLP': 'sum',
            'Clicks': 'sum',
            'Impressions': 'sum',
            'Engagement_Score': 'mean'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        with col1:
            add_tooltip("Tendencia ROI", "Seguimiento mensual del rendimiento promedio de ROI")
            if not monthly_trends.empty:
                # FIXED: Remove markers=True and update traces instead
                fig = px.line(monthly_trends, x='start_date', y='ROI',
                             title="<b>Tendencia Mensual de ROI Promedio</b>",
                             template="plotly_dark")
                fig.update_traces(mode='lines+markers')  # Add markers here
                fig.update_layout(height=400, font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar tendencias de ROI")
        
        with col2:
            add_tooltip("Tendencia de Gasto", "Seguimiento mensual del gasto total en marketing")
            if not monthly_trends.empty:
                # FIXED: Remove markers=True and update traces instead
                fig = px.line(monthly_trends, x='start_date', y='Costo_Adquisicion_CLP',
                             title="<b>Tendencia Mensual de Gasto (CLP)</b>",
                             template="plotly_dark")
                fig.update_traces(mode='lines+markers')  # Add markers here
                fig.update_layout(height=400, font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar tendencias de gasto")
        
        add_tooltip("Rendimiento por Canal", "Compara tendencias de ROI entre diferentes canales de marketing")
        
        channel_trends = filtered_df.groupby(['Channel_Used', pd.Grouper(key='start_date', freq='M')]).agg({
            'ROI': 'mean',
            'Clicks': 'sum'
        }).reset_index()
        
        if not channel_trends.empty:
            # FIXED: Remove markers=True and update traces instead
            fig = px.line(channel_trends, x='start_date', y='ROI',
                         color='Channel_Used',
                         title="<b>Tendencias de ROI por Canal</b>",
                         template="plotly_dark")
            fig.update_traces(mode='lines+markers')  # Add markers here
            fig.update_layout(height=500, font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para mostrar tendencias por canal")
    else:
        st.warning("No hay datos disponibles para análisis de tendencias")

# --------------------------
# Data Explorer Section
# --------------------------
st.markdown("## 🔍 Explorador de Datos")
with st.expander("Filtrado Avanzado", expanded=False):
    if EXTRAS_AVAILABLE:
        try:
            explorer_df = filtered_df.copy()
            explorer_df['start_date'] = explorer_df['start_date'].dt.strftime('%Y-%m-%d')
            explorer_df['end_date'] = explorer_df['end_date'].dt.strftime('%Y-%m-%d')
            
            add_tooltip("Explorador de Datos", "Explorador interactivo con capacidades de filtrado")
            filtered_data = extras['dataframe_explorer'](
                explorer_df, 
                case=False,
                date_format='%Y-%m-%d'
            )
            st.dataframe(filtered_data, use_container_width=True, height=400)
        except Exception as e:
            st.warning(f"Error en explorador avanzado: {str(e)}")
            st.dataframe(filtered_df, use_container_width=True, height=400)
    else:
        st.dataframe(filtered_df, use_container_width=True, height=400)

# CSV Download Functionality
add_tooltip("Descargar Datos", "Descarga los datos filtrados como archivo CSV")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Descargar Datos Filtrados (CSV)",
    data=csv,
    file_name=f"datos_marketing_filtrados_{datetime.now().strftime('%Y%m%d')}.csv",
    mime='text/csv',
    use_container_width=True
)

# --------------------------
# Footer Section
# --------------------------
st.markdown("---")
st.markdown("""
    <div style="text-align: center; font-size: 0.9em; opacity: 0.8;">
    <p>Desarrollado por Bo Kolstrup | Científico de Datos</p>
    <p>Contacto: bokolstrup@gmail.com | +56 9 4259 6282</p>
    <p>Última actualización: {date}</p>
    </div>
""".format(date=datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)