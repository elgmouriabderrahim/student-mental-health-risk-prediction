FROM apache/airflow:3.4.0-python3.12

USER root

COPY requirements.txt /tmp/requirements.txt
RUN sed '/^apache-airflow/d' /tmp/requirements.txt > /tmp/requirements-project.txt

USER airflow

RUN pip install --no-cache-dir -r /tmp/requirements-project.txt

WORKDIR /opt/airflow

COPY --chown=airflow:root src/ /opt/airflow/src/
COPY --chown=airflow:root dag/ /opt/airflow/dags/
COPY --chown=airflow:root app/ /opt/airflow/app/
COPY --chown=airflow:root data/raw/ /opt/airflow/data/raw/
COPY --chown=airflow:root models/ /opt/airflow/models/

ENV PYTHONPATH=/opt/airflow
