import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Tuple


def _find_target_column(columns):
    # Prefer common target names (case-insensitive) or any column containing 'phish', 'result', 'label', 'target'
    candidates = [c for c in columns if c.lower() in ('result', 'label', 'target', 'phishing', 'phising')]
    if candidates:
        return candidates[0]

    for c in columns:
        low = c.lower()
        if 'phish' in low or 'result' in low or 'label' in low or 'target' in low:
            return c

    return None


def load_and_preprocess_data(path: str = None) -> Tuple:
    """Load CSV, detect the target column, preprocess features and return train/test splits.

    Args:
        path: Path to CSV file. If None, will look for data/Phising_Detection_Dataset.csv relative to repo root.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # default path relative to repo root
    if path is None:
        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / 'data' / 'Phising_Detection_Dataset.csv'
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Please place the CSV in the data/ directory.")

    data = pd.read_csv(path)

    # drop unnamed index column if present
    unnamed_cols = [c for c in data.columns if c.startswith('Unnamed')]
    if unnamed_cols:
        data = data.drop(columns=unnamed_cols)

    # Detect target column
    target_col = _find_target_column(list(data.columns))
    if target_col is None:
        raise KeyError(f"Could not detect target column. Available columns: {list(data.columns)}")

    # Raw target and features
    y_raw = data[target_col]
    X = data.drop(columns=[target_col])

    # Coerce target to numeric and map -1 -> 0
    y_numeric = pd.to_numeric(y_raw, errors='coerce')
    y_numeric = y_numeric.replace({-1: 0})

    # Drop rows where target is missing or non-finite
    mask = y_numeric.notna()
    if mask.sum() == 0:
        raise ValueError('No valid target values found after coercion')

    X = X.loc[mask].reset_index(drop=True)
    y_numeric = y_numeric.loc[mask].astype(int).reset_index(drop=True)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_numeric, test_size=0.2, random_state=42
    )

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), scaler
