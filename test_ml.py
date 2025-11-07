import os
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# same categorical features used in train_model.py
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def test_compute_model_metrics_known_values():
    """Metrics should match expected values on a tiny, known example."""
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])
    precision, recall, f1 = compute_model_metrics(y, preds)
    # precision = 1 TP / 1 predicted positive = 1.0
    # recall    = 1 TP / 2 actual positives  = 0.5
    # f1        = 2*(1*0.5)/(1+0.5) ≈ 0.6667
    assert precision == 1.0
    assert recall == 0.5
    assert f1 == pytest.approx(2 * (1 * 0.5) / (1 + 0.5), rel=1e-6)

def test_train_model_returns_random_forest():
    """train_model should return a RandomForestClassifier instance."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_end_to_end_small_sample():
    """Load small data slice, process, train, and ensure predictions are valid."""
    project_path = os.getcwd()
    data_path = os.path.join(project_path, "data", "census.csv")
    df = pd.read_csv(data_path)

    # small, fast sample for CI speed
    df_small = df.sample(n=200, random_state=42)

    train_df, test_df = train_test_split(
        df_small, test_size=0.2, random_state=42, stratify=df_small["salary"]
    )

    X_train, y_train, encoder, lb = process_data(
        train_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    # basic validity checks
    assert len(preds) == len(y_test)
    assert set(np.unique(preds)).issubset({0, 1})
    assert encoder is not None and lb is not None
