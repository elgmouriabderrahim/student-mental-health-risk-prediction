"""Reusable training and evaluation functions for the Airflow DAG."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.transformers import DropColumns, IQRClipper, ScreenTimeConsistency


RANDOM_STATE = 42
CLASS_NAMES = ["Low", "Medium", "High"]
RISK_MAPPING = {"Low": 0, "Medium": 1, "High": 2}

NUMERIC_COLUMNS = [
    "Age",
    "Year_of_Study",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate",
]

ORDINAL_COLUMNS = [
    "Physical_Activity",
    "Diet_Quality",
    "Part_Time_Job",
    "Previous_Mental_Health_History",
]

NOMINAL_COLUMNS = ["Gender"]

CLIP_COLUMNS = [
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Attendance_Rate",
]

REQUIRED_COLUMNS = ["Student_ID", *NUMERIC_COLUMNS, *ORDINAL_COLUMNS, *NOMINAL_COLUMNS]


def create_preprocessor() -> ColumnTransformer:
    """Create the transformations used inside the model pipeline."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    ordinal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[
                        ["Low", "Moderate", "High"],
                        ["Poor", "Average", "Good"],
                        ["No", "Yes"],
                        ["No", "Yes"],
                    ]
                ),
            ),
        ]
    )

    nominal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numerical", numeric_pipeline, NUMERIC_COLUMNS),
            ("ordinal", ordinal_pipeline, ORDINAL_COLUMNS),
            ("nominal", nominal_pipeline, NOMINAL_COLUMNS),
        ]
    )


def create_pipeline() -> Pipeline:
    """Create the full preprocessing and Logistic Regression pipeline."""
    return Pipeline(
        [
            ("drop_identifier", DropColumns(["Student_ID"])),
            ("screen_time_check", ScreenTimeConsistency()),
            ("outlier_clipper", IQRClipper(CLIP_COLUMNS)),
            ("preprocessor", create_preprocessor()),
            ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]
    )


def prepare_data(raw_path: str, output_dir: str) -> dict[str, int]:
    """Load the raw data and save the same stratified 80/20 split as étape 2."""
    raw_data = pd.read_csv(raw_path)
    X = raw_data.drop(columns="Mental_Health_Risk")
    y = raw_data["Mental_Health_Risk"].map(RISK_MAPPING)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output / "X_train.csv", index=False)
    X_test.to_csv(output / "X_test.csv", index=False)
    y_train.to_csv(output / "y_train.csv", index=False)
    y_test.to_csv(output / "y_test.csv", index=False)

    return {"train_rows": len(X_train), "test_rows": len(X_test)}


def validate_preprocessing_inputs(processed_dir: str, report_path: str) -> dict[str, int]:
    """Validate the inputs that will be transformed inside the training pipeline."""
    processed = Path(processed_dir)
    X_train = pd.read_csv(processed / "X_train.csv")
    X_test = pd.read_csv(processed / "X_test.csv")

    missing_columns = set(REQUIRED_COLUMNS) - set(X_train.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    report = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_missing_values": int(X_train.isna().sum().sum()),
        "test_missing_values": int(X_test.isna().sum().sum()),
    }

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def train_model(processed_dir: str, model_path: str, training_report_path: str) -> dict:
    """Tune and save the full model pipeline using training data only."""
    processed = Path(processed_dir)
    X_train = pd.read_csv(processed / "X_train.csv")
    y_train = pd.read_csv(processed / "y_train.csv").squeeze("columns")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=create_pipeline(),
        param_grid={
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__class_weight": [None, "balanced"],
        },
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, model_file)

    report = {
        "model": "Logistic Regression",
        "best_cv_f1_macro": round(float(search.best_score_), 4),
        "best_parameters": search.best_params_,
    }
    Path(training_report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_model(processed_dir: str, model_path: str, metrics_path: str, matrix_path: str) -> dict:
    """Evaluate the saved model once on the independent test set."""
    processed = Path(processed_dir)
    X_test = pd.read_csv(processed / "X_test.csv")
    y_test = pd.read_csv(processed / "y_test.csv").squeeze("columns")
    pipeline = joblib.load(model_path)

    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model": "Logistic Regression",
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(
            float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4
        ),
        "recall_macro": round(
            float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4
        ),
        "f1_macro": round(
            float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4
        ),
        "roc_auc_ovr_macro": round(
            float(roc_auc_score(y_test, y_probability, multi_class="ovr", average="macro")), 4
        ),
        "high_risk_recall": round(float(report["High"]["recall"]), 4),
        "classification_report": report,
    }

    metrics_file = Path(metrics_path)
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(matrix_path)
    return metrics
