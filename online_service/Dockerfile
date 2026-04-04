FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	ONLINE_SERVICE_CONFIG_PATH=/app/scripts/config/config.yaml

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt

COPY scripts ./scripts

EXPOSE 18080

CMD ["python", "scripts/main.py"]
