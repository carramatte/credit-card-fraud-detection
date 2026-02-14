import streamlit as st
import pandas as pd
import requests
import time
import os

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("Credit Card Fraud Detection System")
st.markdown("Upload a CSV file containing transaction data to detect fraudulent activities.")

# Sidebar
st.sidebar.header("Configuration")
default_api_url = os.environ.get("API_URL", "http://127.0.0.1:8000")
api_url = st.sidebar.text_input("API URL", default_api_url)
BATCH_SIZE = st.sidebar.slider("Batch size (rows per request)", 100, 1000, 500, step=100)

# --- Helper functions ---

def wake_up_api(url, max_retries=5):
    """Send a GET request to wake up the API from Render's cold start."""
    for attempt in range(max_retries):
        try:
            r = requests.get(f"{url}/", timeout=30)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)
    return False


def predict_batch(url, batch_data, max_retries=4):
    """Send a batch to the API with retry + exponential backoff for 429/5xx errors."""
    payload = {"data": batch_data}
    for attempt in range(max_retries):
        try:
            response = requests.post(f"{url}/predict", json=payload, timeout=60)
            if response.status_code == 200:
                return response.json(), None
            elif response.status_code == 429:
                wait = 2 ** (attempt + 1)  # 2, 4, 8, 16 seconds
                time.sleep(wait)
                continue
            elif response.status_code >= 500:
                time.sleep(3)
                continue
            else:
                return None, f"Error {response.status_code}: {response.text}"
        except requests.exceptions.ConnectionError:
            time.sleep(3)
            continue
        except Exception as e:
            return None, str(e)
    return None, "Max retries exceeded (429 - Too Many Requests). Try again in a few minutes or reduce batch size."


# --- Main UI ---

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())
        st.info(f"Loaded **{len(df):,}** transactions. They will be processed in batches of {BATCH_SIZE}.")

        if st.button("Analyze Transactions"):
            # Step 1: Wake up the API
            with st.spinner("Connecting to API (this may take up to 30s if the service is starting)..."):
                alive = wake_up_api(api_url)
                if not alive:
                    st.error("Could not reach the API after several attempts. Please verify the API is deployed and try again.")
                    st.stop()

            st.success("API is online!")

            # Step 2: Process in batches
            all_preds = []
            all_probs = []
            records = df.to_dict(orient="records")
            total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

            progress_bar = st.progress(0, text="Processing batches...")
            error_occurred = False

            for i in range(0, len(records), BATCH_SIZE):
                batch = records[i : i + BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1

                progress_bar.progress(
                    batch_num / total_batches,
                    text=f"Processing batch {batch_num}/{total_batches}..."
                )

                result, error = predict_batch(api_url, batch)

                if error:
                    st.error(f"Failed on batch {batch_num}: {error}")
                    error_occurred = True
                    break

                all_preds.extend(result["predictions"])
                all_probs.extend(result["probabilities"])

                # Small pause between batches to avoid rate limiting
                if batch_num < total_batches:
                    time.sleep(1)

            if not error_occurred:
                progress_bar.progress(1.0, text="Done!")

                # Add results to dataframe
                df["Prediction"] = all_preds
                df["Fraud Probability"] = all_probs

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
