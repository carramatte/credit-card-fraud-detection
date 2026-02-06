from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Fraud Detection API")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../models/random_forest_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '../models/scaler.pkl')

model = None
scaler = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")
            
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"Scaler loaded from {SCALER_PATH}")
        else:
            print(f"Warning: Scaler not found at {SCALER_PATH}")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

class TransactionList(BaseModel):
    # We expect a list of records, where each record is a dict of features
    # This allows flexible input, but we assume the caller provides correct columns
    data: list

@app.post("/predict")
def predict(transactions: TransactionList):
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model or Scaler not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(transactions.data)
        
        # Ensure required columns for scaling exist
        cols_to_scale = ['Time', 'Amount']
        if not all(col in df.columns for col in cols_to_scale):
             raise HTTPException(status_code=400, detail="Input must contain 'Time' and 'Amount' fields")
        
        # Scale
        # Note: We must operate on a copy or the original
        # The scaler was fitted on training data, so we use transform
        df_scaled = df.copy()
        df_scaled[cols_to_scale] = scaler.transform(df[cols_to_scale])
        
        # Ensure column order matches model (if necessary, but sklearn usually handles it if names match? 
        # Actually random forest doesn't verify names if passed numpy array, but if passed dataframe it might.
        # But `predict` usually converts to numpy array.
        # Ideally we should enforce order. For now, trusting input or simple alignment.)
        # Let's get the feature names from the model if possible, or assume V1..V28 + Time + Amount
        
        # Predict
        predictions = model.predict(df_scaled)
        probabilities = model.predict_proba(df_scaled)[:, 1]
        
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "active", "model_loaded": model is not None, "scaler_loaded": scaler is not None}
