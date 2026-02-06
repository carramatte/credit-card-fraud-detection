import pandas as pd
import os

def load_data(filepath=None):
    """
    Loads the credit card fraud dataset.
    
    Args:
        filepath (str, optional): Path to the CSV file. If None, resolves automatically relative to project root.
        
    Returns:
        pd.DataFrame: Loaded dataset or None if file not found.
    """
    if filepath is None:
        # Get absolute path to this script (src/load_data.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to project root
        project_root = os.path.dirname(current_dir)
        filepath = os.path.join(project_root, "data", "raw", "creditcard.csv")

    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        print("Please ensure the Kaggle dataset 'creditcard.csv' is in 'data/raw/'")
        return None
        
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    # Test loading
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "raw", "creditcard.csv")
    load_data(data_path)
