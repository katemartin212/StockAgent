FROM python:3.13-slim

WORKDIR /app

# Install system deps for bcrypt and other native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persistent volume mount point for auth.db and cache.db
RUN mkdir -p /data

ENV PYTHONUNBUFFERED=1
ENV AUTH_DB_PATH=/data/auth.db
ENV CACHE_DB_PATH=/data/cache.db

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
