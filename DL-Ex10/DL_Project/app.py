import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler

st.title("🔐 Phishing Website Detection using Deep Learning")

model = load_model(r"E:\DL_Project\models\phishing_model.h5")

# The training dataset has 9 numeric features; collect 9 inputs here
NUM_FEATURES = 9
input_values = []
for i in range(NUM_FEATURES):
    val = st.number_input(f"Enter feature {i+1}", value=0.0)
    input_values.append(val)

if st.button("Predict"):
    input_data = np.array([input_values], dtype=float)
    # Load scaler if available and apply scaling
    try:
        with open(r"E:\DL_Project\models\scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        input_data = scaler.transform(input_data)
    except Exception:
        # If no scaler available, warn the user and continue with raw inputs
        st.warning('Scaler not found — prediction will use unscaled inputs')

    # Ensure proper shape matches model input dim
    if input_data.shape[1] != NUM_FEATURES:
        st.error(f'Invalid number of features. Expected {NUM_FEATURES}, got {input_data.shape[1]}')
    else:
        prediction = model.predict(input_data)
        prob = prediction.ravel()[0]
        result = "🚨 Phishing Website" if prob > 0.5 else "✅ Legitimate Website"
        st.success(f"{result} (probability: {prob:.3f})")
