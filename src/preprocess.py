import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def preprocess_data(df, target_col='Class', test_size=0.2, random_state=42):
    """
    Splits the data and scales Amount and Time.
    
    Args:
        df (pd.DataFrame): The full dataframe.
        target_col (str): The name of the target column.
        test_size (float): Proportion of test set.
        random_state (int): Seed for reproducibility.
        
    Returns:
        X_train, X_test, y_train, y_test (pd.DataFrame/Series)
    """
    if df is None:
        return None, None, None, None

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split first to avoid data leakage during scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale 'Amount' and 'Time'
    # RobustScaler is less prone to outliers
    scaler = RobustScaler()
    
    cols_to_scale = ['Time', 'Amount']
    
    X_train.loc[:, cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test.loc[:, cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train, X_test, y_train, y_test, scaler

def get_resampled_data(X_train, y_train, method='SMOTE', random_state=42):
    """
    Applies resampling to the training data.
    
    Args:
        X_train, y_train: Training data.
        method (str): 'SMOTE', 'Undersampling', or None.
        
    Returns:
        X_resampled, y_resampled
    """
    if method == 'SMOTE':
        sampler = SMOTE(random_state=random_state)
    elif method == 'Undersampling':
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        return X_train, y_train
    
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled
