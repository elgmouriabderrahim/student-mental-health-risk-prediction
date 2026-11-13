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


def prediction_form():
    """Render a form that sends one new student record to the saved pipeline."""
    st.subheader("Test a new student")
    if not MODEL_PATH.exists():
        st.warning("The saved model is missing. Run the Airflow DAG first.")
        return

    with st.form("prediction_form"):
        first_column, second_column = st.columns(2)

        with first_column:
            student_id = st.text_input("Student ID", value="NEW001")
            age = st.number_input("Age", min_value=15, max_value=100, value=21)
            year_of_study = st.number_input("Year of study", min_value=1, max_value=5, value=2)
            sleep_hours = st.number_input("Sleep hours", min_value=0.0, max_value=24.0, value=7.0)
            study_hours = st.number_input("Study hours per week", min_value=0.0, max_value=80.0, value=20.0)
            gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.0)
            social_media_hours = st.number_input(
                "Social media hours", min_value=0.0, max_value=24.0, value=3.0
            )
            screen_time_hours = st.number_input(
                "Screen time hours", min_value=0.0, max_value=24.0, value=6.0
            )

        with second_column:
            gender = st.selectbox("Gender", ["Female", "Male", "Non-binary"])
            physical_activity = st.selectbox("Physical activity", ["Low", "Moderate", "High"])
            financial_stress = st.number_input(
                "Financial stress", min_value=1.0, max_value=10.0, value=5.0
            )
            family_support = st.number_input(
                "Family support", min_value=1.0, max_value=10.0, value=5.0
            )
            attendance_rate = st.number_input(
                "Attendance rate", min_value=0.0, max_value=100.0, value=80.0
            )
            part_time_job = st.selectbox("Part-time job", ["No", "Yes"])
            diet_quality = st.selectbox("Diet quality", ["Poor", "Average", "Good"])
            previous_history = st.selectbox(
                "Previous mental health history", ["No", "Yes"]
            )

        submitted = st.form_submit_button("Predict risk")

    if submitted:
        student = pd.DataFrame(
            [
                {
                    "Student_ID": student_id,
                    "Age": age,
                    "Gender": gender,
                    "Year_of_Study": year_of_study,
                    "Sleep_Hours": sleep_hours,
                    "Study_Hours_Per_Week": study_hours,
                    "GPA": gpa,
                    "Social_Media_Hours": social_media_hours,
                    "Screen_Time_Hours": screen_time_hours,
                    "Physical_Activity": physical_activity,
                    "Financial_Stress": financial_stress,
                    "Family_Support": family_support,
                    "Attendance_Rate": attendance_rate,
                    "Part_Time_Job": part_time_job,
                    "Diet_Quality": diet_quality,
                    "Previous_Mental_Health_History": previous_history,
                }
            ]
        )

        model = load_model()
        predicted_class = int(model.predict(student)[0])
        probabilities = model.predict_proba(student)[0]

        st.success(f"Predicted mental-health risk: {CLASS_NAMES[predicted_class]}")
        probability_table = pd.DataFrame(
            {"Risk level": CLASS_NAMES, "Probability": probabilities}
        ).set_index("Risk level")
        st.bar_chart(probability_table)


def main():
    st.set_page_config(page_title="Student Mental Health Risk", layout="wide")
    st.title("Student Mental Health Risk Prediction")
    st.write("Monitor the trained model and test predictions for new student data.")

    performance_tab, prediction_tab = st.tabs(["Model performance", "New prediction"])
    with performance_tab:
        show_model_performance()
    with prediction_tab:
        prediction_form()


if __name__ == "__main__":
    main()
