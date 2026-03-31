FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libgeos-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only torch BEFORE requirements to prevent yolov5/vietocr from pulling CUDA build (~600MB saved)
RUN pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir vietocr --no-deps

# --- Production stage ---
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgeos-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . /app

# Compile .py to .pyc to protect source code
RUN python -m compileall -b sources/ && \
    find sources/ -name "*.py" ! -name "__init__.py" -delete

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser && \
    mkdir -p sources/Database/uploads sources/static/results sources/static/face && \
    chown -R appuser:appuser /app

USER appuser

ENV ENV=production
EXPOSE 8080

CMD ["python", "run.py"]
