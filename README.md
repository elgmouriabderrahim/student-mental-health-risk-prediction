# Student Mental Health Risk Prediction

An end-to-end machine-learning project that predicts a university student's mental-health risk level: **Low**, **Medium**, or **High**.

> The project uses a simulated, self-reported dataset. It is a learning project and a screening aid only; it must not be used as a medical diagnosis or as the sole basis for decisions about a student.

## Project objective

The goal is to predict the mental-health risk level of a student from academic, behavioral, lifestyle, and psychosocial information.

## Dataset

The raw dataset is stored at `data/raw/student_mental_health.csv`.

| Property | Value |
|---|---|
| Rows | 2,000 |
| Input features | 16 |
| Target | `Mental_Health_Risk` |
| Classes | Low, Medium, High |

## Project workflow

The project includes exploratory analysis, data preparation, model comparison, optimization, Airflow orchestration, and a Streamlit dashboard.
