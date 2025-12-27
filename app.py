import streamlit as st
import numpy as np
import pickle as pk
import pandas as pd

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(
    page_title="Gold Price Prediction",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Gold Price Prediction App")


# =======================
# LOAD MODELS
# =======================
@st.cache_resource
def load_models():
    with open("model.pkl", "rb") as f:
        rf_model = pk.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pk.load(f)

    return rf_model, scaler


rf_model, scaler = load_models()

# =======================
# MODEL SELECTION
# =======================


# =======================
# USER INPUTS
# =======================
st.subheader("Enter Market Values")

EUR_USD = st.number_input("EUR/USD", value=1.10, step=0.01)
SPX = st.number_input("S&P 500 (SPX)", value=4500.0, step=10.0)
USO = st.number_input("US Oil Fund (USO)", value=70.0, step=1.0)
SLV = st.number_input("Silver ETF (SLV)", value=25.0, step=0.5)

year = st.number_input("Year", value=2025, step=1)
month = st.number_input("Month", min_value=1, max_value=12, value=1)
day = st.number_input("Day", min_value=1, max_value=31, value=1)

# =======================
# PREDICTION
# =======================
if st.button("🔮 Predict Gold Price"):
    input_data = np.array([[EUR_USD, SPX, USO, SLV, year, month, day]])

    input_scaled = scaler.transform(input_data)

    prediction = rf_model.predict(input_scaled)

    st.success(f"💰 Predicted Gold Price (GLD): **{prediction[0]:.2f}**")

# =======================
# FOOTER
# =======================
st.markdown("---")
st.caption("Built with ❤️ using Machine Learning & Streamlit")


