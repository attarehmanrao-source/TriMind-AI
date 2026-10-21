# src/hr.py
"""Room 3 - HR Analytics & Employee Attrition Prediction (binary classification).

Pipeline overview:
    load_data -> clean_data -> split -> train_all -> evaluate -> save_model

All public function names from the original module are preserved, so existing
pages keep working. ``save_model`` still stores the bare sklearn Pipeline, so
``joblib.load`` on the saved file behaves exactly as before.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TARGET = "Attrition"
REQUIRED_COLS = ["Age", "MonthlyIncome", "JobSatisfaction", "OverTime", TARGET]
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Paths are anchored to the project root (parent of src/), not to the current
# working directory, so they stay correct wherever Streamlit is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "hr_sample.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# Identifier-like columns carry no predictive signal and only add noise.
ID_LIKE_COLS = {"employeenumber", "employeeid", "employee_id", "empid", "id"}

_TARGET_MAP = {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}


# --------------------------------------------------------------------------- #
# Data loading & cleaning
# --------------------------------------------------------------------------- #
def _find_target(df: pd.DataFrame) -> str:
    """Return the actual name of the attrition column (case/space-insensitive).

    Raises:
        ValueError: if no attrition column exists.
    """
    for col in df.columns:
        if str(col).strip().lower() == TARGET.lower():
            return col
    raise ValueError(
        f"Target column '{TARGET}' not found. "
        f"Available columns: {', '.join(map(str, df.columns))}"
    )


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the HR dataset from CSV (defaults to data/hr_sample.csv)."""
    return pd.read_csv(DEFAULT_DATA_PATH if path is None else path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean raw HR data, returning a model-ready DataFrame.

    Steps: normalise the target column, blank -> NaN, drop rows missing key
    fields, encode the target as 0/1, and drop ID-like and constant columns.

    Raises:
        ValueError: if the target is missing, no rows remain, or only one
            class is present after cleaning.
    """
    df = df.copy()
    df = df.rename(columns={_find_target(df): TARGET})
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.dropna(subset=[c for c in REQUIRED_COLS if c in df.columns])

    df[TARGET] = (df[TARGET].astype(str).str.strip().str.lower().map(_TARGET_MAP))
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    if df.empty:
        raise ValueError("No usable rows left after cleaning the data.")
    if df[TARGET].nunique() < 2:
        raise ValueError("Attrition column must contain both classes (Yes and No).")

    id_cols = [c for c in df.columns
               if c != TARGET and str(c).strip().lower() in ID_LIKE_COLS]
    const_cols = [c for c in df.columns
                  if c != TARGET and df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=sorted(set(id_cols + const_cols)))
    return df.reset_index(drop=True)


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Split a cleaned DataFrame into features, target, numeric and categorical columns."""
    X = df.drop(columns=[TARGET])
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return X, df[TARGET], num, cat


def split(df: pd.DataFrame):
    """Stratified train/test split. Returns (Xt, Xe, yt, ye, num, cat)."""
    X, y, num, cat = get_xy(df)
    Xt, Xe, yt, ye = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    return Xt, Xe, yt, ye, num, cat


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def get_models() -> dict[str, Any]:
    """Return the candidate models, keyed by display name.

    Note: MLPClassifier has no ``class_weight`` option, so on strongly
    imbalanced data its recall can be lower than the other two models.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE),
    }


def _build_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    """Create a fresh preprocessing transformer (scale numeric, one-hot categorical)."""
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])


def train_all(X_train, y_train, num: list[str], cat: list[str]) -> dict[str, Pipeline]:
    """Fit every candidate model and return the fitted pipelines by name.

    Each pipeline gets its own preprocessor (via ``clone``) so the models never
    share, or overwrite, fitted transformer state.
    """
    prep_template = _build_preprocessor(num, cat)
    pipes: dict[str, Pipeline] = {}
    for name, model in get_models().items():
        pipe = Pipeline([("prep", clone(prep_template)), ("model", model)])
        pipe.fit(X_train, y_train)
        pipes[name] = pipe
    return pipes


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(pipes: dict[str, Pipeline], X_test, y_test):
    """Score every pipeline on the test set.

    Returns:
        (metrics_df sorted by F1 descending, dict of predicted probabilities
        per model - ``None`` where probabilities are unavailable).
    """
    rows, probas = [], {}
    for name, pipe in pipes.items():
        pred = pipe.predict(X_test)
        try:
            proba = pipe.predict_proba(X_test)[:, 1]
            auc = round(float(roc_auc_score(y_test, proba)), 4)
        except (AttributeError, ValueError):
            # No predict_proba, or AUC undefined (e.g. single class in y_test).
            proba, auc = None, None
        probas[name] = proba
        rows.append({
            "Model": name,
            "Accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "Precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "F1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "ROC-AUC": auc,
        })
    df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return df, probas


def confusion_df(y_true, y_pred) -> pd.DataFrame:
    """Labelled 2x2 confusion matrix (always 2x2, even if a class is missing)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["Actual: Stayed", "Actual: Left"],
        columns=["Predicted: Stayed", "Predicted: Left"],
    )


def satisfaction_by_attrition(df: pd.DataFrame) -> pd.DataFrame | None:
    """Mean satisfaction-type scores per attrition class (rows = features)."""
    cols = [c for c in df.columns
            if ("satisfaction" in c.lower() or "environment" in c.lower())
            and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return None
    return df.groupby(TARGET)[cols].mean().T


# --------------------------------------------------------------------------- #
# Prediction-form helpers
# --------------------------------------------------------------------------- #
def feature_defaults(df: pd.DataFrame) -> dict[str, Any]:
    """Default input values: median for numeric, mode for categorical features."""
    X, _, num, cat = get_xy(df)
    defaults: dict[str, Any] = {c: float(X[c].median()) for c in num}
    defaults.update({c: str(X[c].mode()[0]) for c in cat})
    return defaults


def category_options(df: pd.DataFrame) -> dict[str, list[str]]:
    """Sorted unique values for each categorical feature (for select boxes)."""
    X, _, _, cat = get_xy(df)
    return {c: sorted(X[c].astype(str).unique().tolist()) for c in cat}


def build_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """Schema information to store next to a saved model (JSON-serialisable)."""
    X, _, num, cat = get_xy(df)
    return {
        "features": X.columns.tolist(),
        "numeric": num,
        "categorical": cat,
        "defaults": feature_defaults(df),
        "category_options": category_options(df),
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_model(
    pipe: Pipeline,
    filename: str = "hr_best_model.pkl",
    metadata: dict[str, Any] | None = None,
    models_dir: str | Path | None = None,
) -> Path:
    """Save a trained pipeline into the models/ folder and return its path.

    The pipeline itself is stored exactly as before. If ``metadata`` is given
    (see ``build_metadata``) it is saved alongside as ``<name>_meta.json``.
    """
    target_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / filename
    joblib.dump(pipe, model_path)

    if metadata is not None:
        meta_path = model_path.with_name(f"{model_path.stem}_meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path


def load_model(
    filename: str = "hr_best_model.pkl",
    models_dir: str | Path | None = None,
) -> Pipeline:
    """Load a pipeline saved by ``save_model``.

    Raises:
        FileNotFoundError: if the model file does not exist yet.
    """
    target_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    model_path = target_dir / filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)