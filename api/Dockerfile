# api/Dockerfile - FastAPI Service
FROM python:3.10-slim

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# API kod
COPY api/ ./api/
COPY config/ ./config/

# Port
EXPOSE 8000

# API başlat
CMD ["uvicorn", "api.predict_service:app", "--host", "0.0.0.0", "--port", "8000"]