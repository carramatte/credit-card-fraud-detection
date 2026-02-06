# Fraud Detection Project

This project implements a credit card fraud detection system using Machine Learning (Random Forest). It includes a training pipeline, a REST API (FastAPI), and a Dashboard (Streamlit).

## Project Structure

- `api/`: FastAPI application for serving the model.
- `dashboard/`: Streamlit dashboard for user interaction.
- `data/`: Directory for datasets (raw and processed).
- `models/`: Where trained models and scalers are saved.
- `notebooks/`: Jupyter notebooks for exploration and training.
- `src/`: Source code for preprocessing and training.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data:**
   Ensure `creditcard.csv` is located in `data/raw/`.

3. **Train Model:**
   ```bash
   python src/train.py
   ```
   This will save `random_forest_model.pkl` and `scaler.pkl` in `models/`.

## Usage

### Run API
```bash
uvicorn api.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

### Run Dashboard
```bash
streamlit run dashboard/app.py
```
The dashboard will open in your browser.

## API Endpoints
- `POST /predict`: Accepts a list of transactions and returns predictions.
- `GET /`: Health check.
