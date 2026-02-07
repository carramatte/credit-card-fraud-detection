import streamlit as st
import pandas as pd
import requests
import json

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("Credit Card Fraud Detection System")
st.markdown("Upload a CSV file containing transaction data to detect fraudulent activities.")

# Sidebar
st.sidebar.header("Configuration")
import os
default_api_url = os.environ.get("API_URL", "http://127.0.0.1:8000")
api_url = st.sidebar.text_input("API URL", default_api_url)

# File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        if st.button("Analyze Transactions"):
            with st.spinner("Analyzing..."):
                # Convert to dict records
                data = df.to_dict(orient="records")
                
                payload = {"data": data}
                
                try:
                    response = requests.post(f"{api_url}/predict", json=payload)
                    
                    if response.status_code == 200:
                        results = response.json()
                        preds = results["predictions"]
                        probs = results["probabilities"]
                        
                        # Add results to dataframe
                        df["Prediction"] = preds
                        df["Fraud Probability"] = probs
                        
                        # Highlight fraud
                        fraud_df = df[df["Prediction"] == 1]
                        
                        st.success("Analysis Complete!")
                        
                        st.metric("Total Transactions", len(df))
                        st.metric("Fraudulent Transactions Detected", len(fraud_df))
                        
                        if not fraud_df.empty:
                            st.warning("Fraudulent Transactions Found!")
                            st.dataframe(fraud_df.style.format({"Fraud Probability": "{:.2%}"}))
                        else:
                            st.info("No fraudulent transactions detected.")
                            
                        # Download results
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Results",
                            csv,
                            "fraud_detection_results.csv",
                            "text/csv",
                            key='download-csv'
                        )
                        
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to API. Please ensure the API is running.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    
    except Exception as e:
        st.error(f"Error reading file: {e}")
