# Data loading and preprocessing pipeline

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import COLUMNS_TO_DROP, TARGET_COLUMN


def load_data(filepath):
    """
    Load the CVD dataset and return features X and target y.

    Parameters
    ----------
    filepath : path to the CSV file (e.g. "data/cvd_dataset.csv")

    Returns
    -------
    X : feature matrix (DataFrame)
    y : target labels (Series)
    """
    df = pd.read_csv(filepath)

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    X = X.drop(columns=COLUMNS_TO_DROP)

    return X, y


def build_preprocessor(X):
    """
    Build a preprocessing pipeline that:
    - Fills missing values
    - Scales numeric features
    - One-hot encodes categorical features

    Parameters
    ----------
    X : feature matrix (used to detect numeric vs categorical columns)

    Returns
    -------
    preprocessor : sklearn ColumnTransformer
    """
    numeric_features     = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer,     numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    return preprocessor