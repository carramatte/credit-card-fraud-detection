import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("Credit Card Fraud Detection System")
st.markdown("Upload a CSV file containing transaction data to detect fraudulent activities.")

# --- Load model and scaler directly (no API needed) ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/random_forest_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.pkl")


@st.cache_resource
def load_model():
    """Load model and scaler once, cached across reruns."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


try:
    model, scaler = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Failed to load model: {e}")

# Sidebar
st.sidebar.header("Status")
if model_loaded:
    st.sidebar.success("Model loaded")
else:
    st.sidebar.error("Model not loaded")

# --- Main UI ---

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())
        st.info(f"Loaded **{len(df):,}** transactions.")

        if st.button("Analyze Transactions"):
            if not model_loaded:
                st.error("Cannot analyze: model is not loaded.")
                st.stop()

            with st.spinner("Analyzing transactions..."):
                # Validate required columns
                cols_to_scale = ["Time", "Amount"]
                if not all(col in df.columns for col in cols_to_scale):
                    st.error("CSV must contain 'Time' and 'Amount' columns.")
                    st.stop()

                # Scale and predict
                df_scaled = df.copy()
                df_scaled[cols_to_scale] = scaler.transform(df[cols_to_scale])

                predictions = model.predict(df_scaled)
                probabilities = model.predict_proba(df_scaled)[:, 1]

            # Add results to dataframe
            df["Prediction"] = predictions
            df["Fraud Probability"] = probabilities

            # Highlight fraud
            fraud_df = df[df["Prediction"] == 1]

            st.success("Analysis Complete!")

            col1, col2 = st.columns(2)
            col1.metric("Total Transactions", f"{len(df):,}")
            col2.metric("Fraudulent Detected", f"{len(fraud_df):,}")

            if not fraud_df.empty:
                st.warning("Fraudulent Transactions Found!")
                st.dataframe(fraud_df.style.format({"Fraud Probability": "{:.2%}"}))
            else:
                st.info("No fraudulent transactions detected.")

            # Download results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results",
                csv,
                "fraud_detection_results.csv",
                "text/csv",
                key="download-csv",
            )

    except Exception as e:
        st.error(f"Error reading file: {e}")
