import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap
from streamlit_shap import st_shap
import warnings
warnings.filterwarnings('ignore')

# ---- Page Configuration ----
st.set_page_config(
    page_title = "Heart Disease Predictor",
    layout= "centered",
    initial_sidebar_state="expanded"
)

# ---- Dark Theme CSS -----

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
        color: #E0E0E0;
    }

    /* Force text color for readability */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stMarkdown p {
        color: #E0E0E0 !important;
    }
    
    /* Muted text for descriptions */
    .muted-text {
        color: #8B949E !important;
    }

    /* Cards/Containers */
    .minimal-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #010409;
        border-right: 1px solid #30363D;
    }
    
    /* Sidebar Font Sizes */
    [data-testid="stSidebar"] h2 {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] .muted-text {
        font-size: 1.1rem !important;
    }
    
    /* Sidebar Toggle Button (Collapse/Expand) */
    [data-testid="stSidebar"] button, [data-testid="collapsedControl"] {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] button svg, [data-testid="collapsedControl"] svg {
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }
    
    /* Widget Labels (Sidebar & Main) */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #FFFFFF !important;
        font-weight: 500;
    }

    /* Input Fields - Standardize Dark Look */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input {
        background-color: #0D1117 !important;
        color: #E0E0E0 !important;
        border: 1px solid #30363D !important;
    }
    
    /* Selectbox is tricky, target the container */
    [data-baseweb="select"] > div {
        background-color: #0D1117 !important;
        color: #E0E0E0 !important;
        border: 1px solid #30363D !important;
    }
    
    /* Number Input Buttons (inc/dec) */
    button[kind="secondary"] {
        background-color: #161B22 !important;
        color: #E0E0E0 !important;
        border: 1px solid #30363D !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] {
        color: #B0B8C1 !important;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }
    
    /* Slider Min/Max & Value adjustments */
    [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {
        color: #E0E0E0 !important;
        font-size: 0.9rem !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #238636;
        color: white !important;
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #2EA043;
        border-color: #8B949E;
    }
    
    /* Primary Button override */
    div[data-testid="stButton"] button {
        background-color: #1F6FEB;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #388BFD;
    }
    
    /* Risk Indicators */
    .risk-high { border-left: 4px solid #F85149; }
    .risk-low { border-left: 4px solid #238636; }

    /* Hide Streamlit Header/Toolbar to fix "white bar" issue */
    header[data-testid="stHeader"] {
        background-color: #0E1117;
    }

</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div style="margin-bottom: 30px;">
    <h1 style="margin-bottom: 8px;">Heart Disease Predictor</h1>
    <p class="muted-text">Advanced AI-Powered Risk Assessment</p>
</div>
""", unsafe_allow_html=True)
