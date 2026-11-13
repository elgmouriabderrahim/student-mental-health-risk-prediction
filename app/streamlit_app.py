"""Streamlit dashboard for model monitoring and individual predictions."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Required before loading the pipeline because it uses these custom classes.
from src import transformers  # noqa: F401


MODEL_PATH = PROJECT_ROOT / "models/mental_health_risk_pipeline.joblib"
METRICS_PATH = PROJECT_ROOT / "models/model_metrics.json"
MATRIX_PATH = PROJECT_ROOT / "models/confusion_matrix.csv"
CLASS_NAMES = ["Low", "Medium", "High"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, encoding="utf-8") as file:
        return json.load(file)


def show_model_performance():
    """Show metrics, a metric chart, the report, and confusion matrix."""
    if not METRICS_PATH.exists() or not MATRIX_PATH.exists():
        st.info("Run the Airflow DAG first to generate the evaluation artifacts.")
        return

    metrics = load_metrics()
    st.subheader("Model performance")
    st.caption(f"Selected model: {metrics['model']}")

    columns = st.columns(5)
    values = [
        ("Accuracy", metrics["accuracy"]),
        ("Precision", metrics["precision_macro"]),
        ("Recall", metrics["recall_macro"]),
        ("F1-score", metrics["f1_macro"]),
        ("ROC-AUC", metrics["roc_auc_ovr_macro"]),
    ]
    for column, (label, value) in zip(columns, values):
        column.metric(label, f"{value:.3f}")

    metric_chart = pd.DataFrame(values, columns=["Metric", "Score"]).set_index("Metric")
    st.bar_chart(metric_chart)
    st.write(f"High-risk recall: {metrics['high_risk_recall']:.3f}")

    st.subheader("Classification report")
    report = pd.DataFrame(metrics["classification_report"]).T
    st.dataframe(report.round(3), use_container_width=True)

    st.subheader("Confusion matrix")
    matrix = pd.read_csv(MATRIX_PATH, index_col=0)
    figure, axis = plt.subplots(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set_xlabel("Predicted risk")
    axis.set_ylabel("Actual risk")
    st.pyplot(figure)


def main():
    st.set_page_config(page_title="Student Mental Health Risk", layout="wide")
    st.title("Student Mental Health Risk Prediction")
    show_model_performance()


if __name__ == "__main__":
    main()
