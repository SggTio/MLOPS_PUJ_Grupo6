FROM apache/airflow:2.9.3

USER root
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# install uv (fast resolver) and use it for system site-packages
USER airflow
COPY requirements.txt /opt/airflow/requirements.txt
RUN pip install --no-cache-dir uv && \
    uv pip install --system -r /opt/airflow/requirements.txt
