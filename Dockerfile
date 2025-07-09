    FROM python:3.11-slim

    # Install build dependencies
    RUN apt-get update && apt-get install -y \
        gcc \
        python3-dev \
        && rm -rf /var/lib/apt/lists/*

    WORKDIR /app
    COPY . .
    RUN pip install --no-cache-dir -r requirements.txt

    EXPOSE 8000
    CMD ["uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "8000"]
