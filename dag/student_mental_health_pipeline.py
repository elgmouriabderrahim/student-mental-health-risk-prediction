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
