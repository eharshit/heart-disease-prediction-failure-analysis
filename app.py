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

