"""Airflow DAG for the student mental-health risk prediction workflow."""

from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_workflow import (
    evaluate_model,
    prepare_data,
    train_model,
    validate_preprocessing_inputs,
)


logger = logging.getLogger(__name__)
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def log_failure(context):
    """Write failed task information to the Airflow task log."""
    task_id = context["task_instance"].task_id
    logger.error("Task '%s' failed: %s", task_id, context.get("exception"))


default_args = {
    "owner": "student-mental-health-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": log_failure,
}


with DAG(
    dag_id="student_mental_health_training",
    description="Prepare data, train, and evaluate the mental-health risk model.",
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * 1",
    catchup=False,
    default_args=default_args,
    tags=["machine-learning", "student-mental-health"],
) as dag:
    prepare_data_task = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data,
        op_kwargs={
            "raw_path": str(DATA_DIR / "raw/student_mental_health.csv"),
            "output_dir": str(PROCESSED_DIR),
        },
    )

    preprocessing_task = PythonOperator(
        # The actual fitting of transformations happens inside train_model's
        # scikit-learn pipeline, preventing data leakage during cross validation.
        task_id="preprocess_data",
        python_callable=validate_preprocessing_inputs,
        op_kwargs={
            "processed_dir": str(PROCESSED_DIR),
            "report_path": str(MODELS_DIR / "preprocessing_report.json"),
        },
    )

    training_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        op_kwargs={
            "processed_dir": str(PROCESSED_DIR),
            "model_path": str(MODELS_DIR / "mental_health_risk_pipeline.joblib"),
            "training_report_path": str(MODELS_DIR / "training_report.json"),
        },
    )

    evaluation_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
        op_kwargs={
            "processed_dir": str(PROCESSED_DIR),
            "model_path": str(MODELS_DIR / "mental_health_risk_pipeline.joblib"),
            "metrics_path": str(MODELS_DIR / "model_metrics.json"),
            "matrix_path": str(MODELS_DIR / "confusion_matrix.csv"),
        },
    )

    prepare_data_task >> preprocessing_task >> training_task >> evaluation_task
