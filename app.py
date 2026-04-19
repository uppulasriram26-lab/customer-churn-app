import os
import joblib
import streamlit as st

st.write("App starting...")  # DEBUG LINE

try:
    model = joblib.load("churn_model.pkl")
    features = joblib.load("features.pkl")
    st.write("Model loaded successfully ✔")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()