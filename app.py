import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.title("🩺 Breast Cancer Prediction App")
st.write("Linear Regression Model (Educational Purpose Only)")

st.warning("⚠️ This app is for learning purposes only, not medical diagnosis.")

# Feature names
features = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

st.subheader("Enter Cell Nuclei Measurements")

input_data = []
cols = st.columns(3)

for i, feature in enumerate(features):
    with cols[i % 3]:
        value = st.number_input(feature, min_value=0.0, format="%.5f")
        input_data.append(value)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")
    st.write(f"Model Output Value: **{prediction:.4f}**")

    if prediction >= 0.5:
        st.error("🧬 Prediction: **Malignant (Cancerous)**")
    else:
        st.success("🧬 Prediction: **Benign (Non-Cancerous)**")
