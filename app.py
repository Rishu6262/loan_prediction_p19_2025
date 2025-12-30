import streamlit as st
import numpy as np
import pandas as pd
import pickle as pk

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(
    page_title="Gold Price Prediction",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Gold Price Prediction App")
st.write("Predict **GLD (Gold ETF Price)** using Machine Learning")

# =======================
# LOAD MODEL & SCALER
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

    input_dict = {
        "EUR_USD": EUR_USD,
        "SPX": SPX,
        "USO": USO,
        "SLV": SLV,
        "year": year,
        "month": month,
        "day": day
    }

    # ==========================
    # ALIGN FEATURES WITH MODEL
    # ==========================
    model_features = rf_model.feature_names_in_

    input_df = pd.DataFrame(
        [[input_dict[feat] for feat in model_features]],
        columns=model_features
    )

    # ==========================
    # SCALE ONLY IF REQUIRED
    # ==========================
    if hasattr(scaler, "feature_names_in_") and len(scaler.feature_names_in_) == len(model_features):
        input_final = scaler.transform(input_df)
    else:
        # Model was trained without scaling
        input_final = input_df.values

    # ==========================
    # PREDICT
    # ==========================
    prediction = rf_model.predict(input_final)

    st.success(
        f"💰 Predicted Gold Price (GLD): **{prediction[0]:.2f}**"
    )

