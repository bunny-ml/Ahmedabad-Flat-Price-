import streamlit as st
import numpy as np
import joblib

st.title('🏠 Flat Price Prediction App')


model = joblib.load('model/xgboost_pipeline_model.pkl')
st.markdoown('Select the property details from below')


categorical_features = {
    'furnishing': ['Semi', 'Full', 'None'],
    'area_type': ['Super built-up', 'Built-up', 'Carpet'],
    'transaction': ['Resale', 'New']
}

numerical_features = {
    'BHK': (1, 10, 1),  # min, max, step
    'size': (300, 5000, 50)  # example numeric input
}

# --- Categorical Inputs ---
inputs = {}
for feature, options in categorical_features.items():
    inputs[feature] = st.selectbox(f"{feature}", options)

# --- Numerical Inputs ---
for feature, (min_val, max_val, step) in numerical_features.items():
    inputs[feature] = st.number_input(f"{feature}", min_value=min_val, max_value=max_val, step=step)

# Convert inputs to dataframe
input_df = pd.DataFrame([inputs])

st.subheader("Input Details")
st.write(input_df)

# Predict button
if st.button("Predict Price"):
    # Predict (model expects original features, pipeline handles preprocessing)
    pred_log = model.predict(input_df)
    pred_price = np.expm1(pred_log)  # inverse of log1p

    st.success(f"🏡 Predicted Price: ₹{pred_price[0]:,.2f}")