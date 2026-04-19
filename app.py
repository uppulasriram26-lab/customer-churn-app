import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

st.title("📊 Customer Churn Prediction App")
st.markdown("### Developed by: Srira")

# Load model
model = joblib.load("churn_model.pkl")
features = joblib.load("features.pkl")

# Sidebar inputs
st.sidebar.header("Customer Profile")

tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly = st.sidebar.slider("Monthly Charges ($)", 18, 120, 65)
total = st.sidebar.number_input("Total Charges ($)", 18, 9000, 500)

# Predict button
if st.button("Calculate Churn Risk"):

    input_dict = {col: 0 for col in features}

    input_dict["tenure"] = tenure
    input_dict["MonthlyCharges"] = monthly
    input_dict["TotalCharges"] = total

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    st.metric("Churn Probability", f"{probability*100:.1f}%")
    st.progress(int(probability * 100))

    if probability > 0.7:
        st.error("🚨 High Risk Customer")
    elif probability > 0.4:
        st.warning("⚠️ Medium Risk Customer")
    else:
        st.success("✅ Low Risk Customer")