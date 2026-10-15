#src/churn.py
"""Room 2 — Customer Churn Prediction (binary classification)."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
REQUIRED_COLS = ["gender", "tenure", "MonthlyCharges", "TotalCharges", TARGET]
RANDOM_STATE = 42


def load_data(path="data/churn_sample.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    for c in ["TotalCharges"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[c for c in REQUIRED_COLS if c in df.columns])
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True)


def get_xy(df: pd.DataFrame):
    drop = [TARGET]
    for c in ["customerID"]:
        if c in df.columns:
            drop.append(c)
    X = df.drop(columns=drop)
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return X, df[TARGET], num, cat


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE),
    }


def train_all(X_train, y_train, num, cat):
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    pipes = {}
    for name, model in get_models().items():
        pipe = Pipeline([("prep", prep), ("model", model)])
        pipe.fit(X_train, y_train)
        pipes[name] = pipe
    return pipes


def evaluate(pipes, X_test, y_test):
    rows, probas = [], {}
    for name, pipe in pipes.items():
        pred = pipe.predict(X_test)
        try:
            proba = pipe.predict_proba(X_test)[:, 1]
            auc = round(float(roc_auc_score(y_test, proba)), 4)
        except Exception:
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


def split(df: pd.DataFrame):
    X, y, num, cat = get_xy(df)
    Xt, Xe, yt, ye = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    return Xt, Xe, yt, ye, num, cat


def feature_defaults(df: pd.DataFrame):
    X, _, num, cat = get_xy(df)
    defaults = {}
    for c in num:
        defaults[c] = float(X[c].median())
    for c in cat:
        defaults[c] = str(X[c].mode()[0])
    return defaults


def category_options(df: pd.DataFrame):
    X, _, _, cat = get_xy(df)
    return {c: sorted(X[c].astype(str).unique().tolist()) for c in cat}


def confusion_df(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, index=["Actual: Stayed", "Actual: Churned"],
                        columns=["Predicted: Stayed", "Predicted: Churned"])