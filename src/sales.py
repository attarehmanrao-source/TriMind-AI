# src/sales.py
"""Room 1 — Sales Analytics & Profit Prediction (regression)."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Profit"
REQUIRED_COLS = ["Segment", "Region", "Category", "Sub-Category", "Ship Mode",
                 "Quantity", "Discount", "Sales", "Profit"]
NUMERIC = ["Quantity", "Discount", "Sales", "OrderYear", "OrderMonth"]
CATEGORICAL = ["Segment", "Region", "Category", "Sub-Category", "Ship Mode"]
RANDOM_STATE = 42


def load_data(path="data/sales_sample.csv") -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin1")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=REQUIRED_COLS)
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
        df["OrderYear"] = df["Order Date"].dt.year
        df["OrderMonth"] = df["Order Date"].dt.month
        df = df.dropna(subset=["OrderYear", "OrderMonth"])
    else:
        df["OrderYear"] = 2023
        df["OrderMonth"] = 6
    df = df[df["Sales"] > 0]
    return df.reset_index(drop=True)


def get_xy(df: pd.DataFrame):
    return df[NUMERIC + CATEGORICAL], df[TARGET]


def _preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
        "Neural Network (MLP)": MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE),
    }


def train_all(X_train, y_train):
    pipes = {}
    for name, model in get_models().items():
        pipe = Pipeline([("prep", _preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        pipes[name] = pipe
    return pipes


def evaluate(pipes, X_test, y_test):
    rows = []
    for name, pipe in pipes.items():
        pred = pipe.predict(X_test)
        rows.append({
            "Model": name,
            "R²": round(r2_score(y_test, pred), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
            "MAE": round(float(mean_absolute_error(y_test, pred)), 2),
        })
    return pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)


def split(df: pd.DataFrame):
    X, y = get_xy(df)
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def feature_defaults(df: pd.DataFrame):
    defaults = {}
    for c in NUMERIC:
        defaults[c] = float(df[c].median())
    for c in CATEGORICAL:
        defaults[c] = str(df[c].mode()[0])
    return defaults


def category_options(df: pd.DataFrame):
    return {c: sorted(df[c].astype(str).unique().tolist()) for c in CATEGORICAL}


# --- NAYA FUNCTION: Model Save Karne Ke Liye ---
def save_model(pipe, filename="sales_best_model.pkl"):
    """Trained model ko models/ folder mein save karta hai."""
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, os.path.join("models", filename))