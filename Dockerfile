# Multi-stage production Dockerfile for AutoSpec AI
# Stage 1: Dependency builder
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create non-root system user and secure directories
RUN groupadd -r autospec && useradd -r -g autospec -d /app autospec && \
    mkdir -p /app/data /app/exports && \
    chown -R autospec:autospec /app

# Copy application code
COPY --chown=autospec:autospec src/ ./src/
COPY --chown=autospec:autospec api/ ./api/
COPY --chown=autospec:autospec app.py pyproject.toml ./

# Install local package in editable mode
RUN pip install --no-cache-dir -e .

USER autospec

# Expose FastAPI port (8000) and Streamlit port (8501)
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: launch FastAPI server
CMD ["uvicorn", "api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
