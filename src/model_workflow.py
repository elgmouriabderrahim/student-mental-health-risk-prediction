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
