# Dockerfile - Training Pipeline
FROM python:3.10-slim

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları (ML için)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kod dosyaları
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY data/ ./data/

# Training çalıştır
CMD ["python", "scripts/train_cleaned_data.py"]