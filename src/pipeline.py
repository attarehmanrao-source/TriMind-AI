# src/pipeline.py
"""Dynamic ML pipeline logic (load, preprocess, train, evaluate)."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, LabelEncoder, FunctionTransformer)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             r2_score, mean_squared_error, mean_absolute_error, confusion_matrix)

ENCODING_OPTIONS = {
    "One-Hot Encoding": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "Label Encoding": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
}

# Yahan "None" ke liye string ki jagah FunctionTransformer use kiya hai taake error na aaye
SCALING_OPTIONS = {
    "Standardization (Z-score)": StandardScaler(),
    "Normalization (Min-Max)": MinMaxScaler(),
    "None": FunctionTransformer(validate=False), 
}

MODELS = {
    "classification": {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=30, max_depth=10, n_jobs=-1, random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=200, random_state=42)
    },
    "regression": {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=30, max_depth=10, n_jobs=-1, random_state=42),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=200, random_state=42)
    }
}

def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file, encoding="latin1")

def drop_columns(df, cols):
    return df.drop(columns=cols, errors="ignore")

def split_column_types(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    txt_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, txt_cols

def missing_summary(df):
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if miss.empty:
        return pd.DataFrame()
    return pd.DataFrame({"Column": miss.index, "Missing Values": miss.values})

def detect_task(target_series):
    # Auto-detect: If less than 15 unique values or text -> Classification
    if target_series.dtype == 'object' or target_series.nunique() < 15:
        return "classification"
    return "regression"

def train_and_evaluate(df, target, model_name, task, encoding, scaling, test_size=0.2):
    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target]

    le = None
    if task == "classification":
        if y.dtype == 'object' or not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
    
    num_cols, txt_cols = split_column_types(X)

    # FIX: Yahan direct `scaling` aur `encoding` use kiya gaya hai (pehle dictionary se dhoondne ki koshish ho rahi thi)
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaling)
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoding)
    ])

    transformers = []
    if num_cols: transformers.append(("num", num_transformer, num_cols))
    if txt_cols: transformers.append(("cat", cat_transformer, txt_cols))
    
    preprocessor = ColumnTransformer(transformers)

    model = MODELS[task][model_name]

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    metrics = {}
    cm = None
    classes = None
    importance = None

    if task == "classification":
        metrics = {
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, preds, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_test, preds, average="weighted", zero_division=0)
        }
        cm = confusion_matrix(y_test, preds)
        classes = [str(c) for c in (le.classes_ if le else np.unique(y))]
    else:
        metrics = {
            "R²": r2_score(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "MAE": mean_absolute_error(y_test, preds)
        }

    # Feature Importance
    model_step = pipe.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        try:
            feature_names = pipe.named_steps["prep"].get_feature_names_out()
            importances = model_step.feature_importances_
            importance = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            importance = importance.sort_values("Importance", ascending=False).reset_index(drop=True)
        except Exception:
            pass

    return {
        "metrics": metrics,
        "y_test": y_test,
        "preds": preds,
        "confusion_matrix": cm,
        "classes": classes,
        "importance": importance,
        "label_encoder": le,
        "model_name": model_name,
        "task": task,
        "target": target,
        "n_train": len(X_train),
        "n_test": len(X_test)
    }