import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os
from preprocess import preprocess_data, get_resampled_data

# Define paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/raw/creditcard.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/random_forest_model.pkl')

def train_model():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Preprocess
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Save scaler
    scaler_path = os.path.join(os.path.dirname(MODEL_PATH), 'scaler.pkl')
    print(f"Saving scaler to {scaler_path}...")
    joblib.dump(scaler, scaler_path)
    
    # Resample (using SMOTE as default from our exploration)
    print("Resampling training data...")
    X_train_res, y_train_res = get_resampled_data(X_train, y_train, method='SMOTE')
    
    # Train
    print("Training Random Forest model...")
    # Using fewer estimators for speed in this demo, increase for production
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_train_res, y_train_res)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Save model
    print(f"Saving model to {MODEL_PATH}...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print("Done!")

if __name__ == "__main__":
    train_model()
