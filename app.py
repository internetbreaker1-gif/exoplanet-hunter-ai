import streamlit as st
import pickle
import numpy as np

# Load model
with open('models/exo_model.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['model']
scaler = data['scaler']
le = data['label_encoder']

# Preloaded example planets
examples = {
    "Kepler-22b": [289.9, 2.38, 0.0, 5518, 1.1, 0.97],       # CONFIRMED
    "Proxima b": [11.2, 1.1, 0.0, 3042, 0.14, 0.12],         # CANDIDATE
    "Fake Planet": [0.05, 15.0, 50.0, 900, 0.01, 0.01],     # FALSE POSITIVE
    "Earth": [365.25, 1.0, 0.00315, 5778, 1.0, 1.0]          # CONFIRMED
}

# Streamlit UI
st.title("ExoHunter AI 🌌")
st.write("Predict if a potential exoplanet is confirmed, candidate, or false positive.")

# Example planet selection
example_name = st.selectbox("Choose an example planet or select 'Custom' to enter values:", ["Custom"] + list(examples.keys()))

if example_name == "Custom":
    # Inputs for custom planet (allow extreme values)
    orbper = st.number_input("Orbital Period (days)", min_value=0.0, value=10.0, step=0.01)
    rade = st.number_input("Planet Radius (Earth radii)", min_value=0.0, value=1.0, step=0.01)
    bmassj = st.number_input("Planet Mass (Jupiter masses)", min_value=0.0, value=0.5, step=0.01)
    teff = st.number_input("Star Temperature (K)", min_value=0, value=5500, step=1)
    srad = st.number_input("Star Radius (Solar radii)", min_value=0.0, value=1.0, step=0.01)
    smass = st.number_input("Star Mass (Solar masses)", min_value=0.0, value=1.0, step=0.01)
else:
    # Load example planet values
    orbper, rade, bmassj, teff, srad, smass = examples[example_name]
    st.write(f"Using example planet **{example_name}** with preloaded values:")
    st.write(f"Orbital Period: {orbper}, Planet Radius: {rade}, Planet Mass: {bmassj}, Star Temp: {teff}, Star Radius: {srad}, Star Mass: {smass}")

# Predict
if st.button("Predict"):
    features = np.array([[orbper, rade, bmassj, teff, srad, smass]])
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)
    pred_label = le.inverse_transform(pred)[0]
    st.success(f"Prediction: **{pred_label}**")
