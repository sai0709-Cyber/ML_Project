import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("beam_model.pkl")

st.title("🔧 Beam Deformation Predictor")
st.markdown("Enter beam dimensions and pressure to predict maximum deformation.")

# Input sliders
length = st.slider("Length (m)", 0.1, 1.0, 0.3)
width = st.slider("Width (m)", 0.01, 0.1, 0.05)
height = st.slider("Height (m)", 0.01, 0.1, 0.05)
pressure = st.slider("Pressure (Pa)", 1e6, 30e6, 5e6, step=1e6)

# Shape selection (if model supports one-hot)
shape = st.selectbox("Shape", ["rect", "sphere"])
shape_sphere = 1 if shape == "sphere" else 0

# Predict
if st.button("Predict"):
    features = np.array([[length, width, height, pressure, shape_sphere]])
    prediction = model.predict(features)[0]
    st.success(f"📈 Predicted Max Deformation: {prediction:.4e} m")
