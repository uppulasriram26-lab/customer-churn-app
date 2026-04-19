import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Srira's Churn Predictor", page_icon="📊")

st.title("📊 Customer Churn Prediction App")
st.markdown("### Developed by: **Srira**")

# Load assets
try:
    model = joblib.load('churn_model.pkl')
    model_features = joblib.load('features.pkl')

    st.sidebar.header("Customer Profile")
    tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
    monthly = st.sidebar.slider("Monthly Charges ($)", 18, 120, 65)
    total = st.sidebar.number_input("Total Charges ($)", 18, 9000, 500)

    if st.button("Calculate Churn Risk"):
        # Create a single row of data with the correct feature names
        input_dict = {col: 0 for col in model_features} # Initialize all 45 features to 0
        
        # Update the specific features from our sliders
        input_dict['tenure'] = tenure
        input_dict['MonthlyCharges'] = monthly
        input_dict['TotalCharges'] = total
        
        # Convert to DataFrame so it has 'valid feature names'
        input_df = pd.DataFrame([input_dict])
        
        # Make Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"🚨 **High Risk!** Probability of Churn: {probability*100:.1f}%")
        else:
            st.success(f"✅ **Low Risk.** Probability of Churn: {probability*100:.1f}%")