import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib

# Load model
model = joblib.load("model/xgboost_pipeline_model.pkl")

# Load feature JSONs
with open("model/categorical_features.json") as f:
    cat_features = json.load(f)

with open("model/numeric_features.json") as f:
    num_features = json.load(f)

st.title("🏠 Flat Price Prediction")

user_input = {}

# Dropdowns for categorical features
for feature, options in cat_features.items():
    user_input[feature] = st.selectbox(f"{feature}", options)

# Numeric inputs
for feature, props in num_features.items():
    # Special case for area_sqft
    if feature == "area_sqft":
        # User sees normal area
        area_real = st.number_input(
            "Area (sq.ft)", 
            min_value=int(np.expm1(props['min'])),  # convert log min back to normal
            max_value=int(np.expm1(props['max'])),  # convert log max back to normal
            value=int(np.expm1(props['mean']))      # convert mean back
        )
        # Transform back to log1p for model
        user_input[feature] = np.log1p(area_real)
    else:
        user_input[feature] = st.number_input(
            feature, 
            min_value=props['min'], 
            max_value=props['max'], 
            value=int(props['mean'])
        )

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

st.subheader("Input Details")
st.write(input_df)

# Predict button
if st.button("Predict Price"):
    pred_log = model.predict(input_df)
    pred_price = np.expm1(pred_log)  # inverse of log1p
    st.success(f"🏡 Predicted Price: ₹{pred_price[0]:,.2f}")
