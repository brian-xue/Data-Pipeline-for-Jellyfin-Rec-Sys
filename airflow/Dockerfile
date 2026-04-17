FROM apache/airflow:2.11.0

USER root
# Install system dependencies if needed (e.g. for psycopg2)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements
COPY requirements.txt /requirements.txt

# Install python dependencies
RUN pip install --no-cache-dir -r /requirements.txt