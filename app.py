import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

st.title("📊 Customer Churn Prediction App")

st.markdown("""
### 🚀 Project Overview
This application predicts customer churn using Machine Learning (Random Forest).
It helps telecom companies identify customers likely to leave.
""")

# =========================
# LOAD MODEL
# =========================
model = joblib.load("churn_model.pkl")
features = joblib.load("features.pkl")

st.success("Model loaded ✔")

# =========================
# FEATURE IMPORTANCE (OUTSIDE BUTTON)
# =========================
if hasattr(model, "feature_importances_"):
    st.subheader("📊 Top Important Features")

    importance = model.feature_importances_
    indices = np.argsort(importance)[-10:]

    fig, ax = plt.subplots()
    ax.barh(range(len(indices)), importance[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([features[i] for i in indices])
    ax.set_title("Feature Importance (Top 10)")

    st.pyplot(fig)

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("Customer Profile")

tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly = st.sidebar.slider("Monthly Charges ($)", 18, 120, 65)
total = st.sidebar.number_input("Total Charges ($)", 18, 9000, 500)

# =========================
# PREDICTION
# =========================
if st.button("Calculate Churn Risk"):

    input_dict = {col: 0 for col in features}

    input_dict["tenure"] = tenure
    input_dict["MonthlyCharges"] = monthly
    input_dict["TotalCharges"] = total

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    st.subheader("📊 Prediction Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Churn Probability", f"{probability*100:.1f}%")

    col2.metric(
        "Risk Level",
        "High" if probability > 0.7 else
        "Medium" if probability > 0.4 else "Low"
    )

    col3.metric("Model", "Random Forest")

    st.progress(int(probability * 100))

    # =========================
    # EXPLANATION
    # =========================
    st.subheader("🧠 Explanation")

    if probability > 0.7:
        st.error("High churn risk due to high charges or low tenure.")
    elif probability > 0.4:
        st.warning("Medium risk. Customer may switch if better offers appear.")
    else:
        st.success("Low risk customer. Likely to stay.")