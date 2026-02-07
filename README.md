# 🛡️ Credit Card Fraud Detection System

A complete **Machine Learning** solution for detecting fraudulent credit card transactions in real-time.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://fraud-detection-dashboard-ml39.onrender.com)
[![API](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fraud-detection-api-zchp.onrender.com)

---

## 🎯 Overview

This project implements an end-to-end fraud detection pipeline using **Random Forest** classification. It includes data preprocessing, model training, a REST API for predictions, and an interactive dashboard for analysis.

## 🚀 Live Demo

- **Dashboard:** [fraud-detection-dashboard-ml39.onrender.com](https://fraud-detection-dashboard-ml39.onrender.com)
- **API:** [fraud-detection-api-zchp.onrender.com](https://fraud-detection-api-zchp.onrender.com)

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML/Data** | Python, Scikit-learn, Pandas, NumPy |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | Streamlit |
| **Deployment** | Render |

## 📁 Project Structure

```
fraud_detection/
├── api/                 # FastAPI REST API
│   └── main.py
├── dashboard/           # Streamlit Dashboard
│   └── app.py
├── src/                 # Training pipeline
│   ├── load_data.py
│   ├── preprocess.py
│   └── train.py
├── models/              # Trained models
├── notebooks/           # Jupyter notebooks
└── data/                # Datasets
```

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python src/train.py
```

### 3. Run the API
```bash
uvicorn api.main:app --reload
```

### 4. Run the Dashboard
```bash
streamlit run dashboard/app.py
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predict fraud on transactions |

## 📝 License

MIT License

---

Made with ❤️ by [@carramatte](https://github.com/carramatte)
