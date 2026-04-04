FROM python:3.11-slim

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
    

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
# RUN pip install --no-cache-dir "numpy<2.0.0" "pyarrow>=14.0.0" "fsspec==2023.9.0"


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/

# CMD ["sh", "-c", "python scripts/build_embedding_text.py && python scripts/emdding.py && python scripts/build_initial_user.py && python scripts/build_offline_samples.py"]