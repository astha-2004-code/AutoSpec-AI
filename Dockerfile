# Multi-stage production Dockerfile for AutoSpec AI 2.0 FastAPI Backend (<512MB Render Free Tier)
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache lightweight INT8 quantized ONNX embedding model (~23 MB, zero PyTorch/CUDA)
ARG PRECACHE_ONNX_REPO=Xenova/all-MiniLM-L6-v2
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('${PRECACHE_ONNX_REPO}', 'onnx/model_quantized.onnx'); hf_hub_download('${PRECACHE_ONNX_REPO}', 'tokenizer.json')" || true

FROM python:3.11-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache/huggingface /home/autospec/.cache/huggingface
ENV PATH="/opt/venv/bin:$PATH"
ENV HF_HOME="/home/autospec/.cache/huggingface"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV MALLOC_TRIM_THRESHOLD_=65536
ENV PORT=8000

RUN groupadd -r autospec && useradd -r -g autospec -d /app autospec && \
    mkdir -p /app/storage /app/data && \
    chown -R autospec:autospec /app /home/autospec

COPY --chown=autospec:autospec src/ ./src/
COPY --chown=autospec:autospec api/ ./api/
COPY --chown=autospec:autospec scripts/ ./scripts/
COPY --chown=autospec:autospec data/ ./data/
COPY --chown=autospec:autospec migrations/ ./migrations/
COPY --chown=autospec:autospec alembic.ini pyproject.toml README.md ./

RUN pip install --no-cache-dir --no-deps -e .

USER autospec

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD sh -c "curl -f http://localhost:${PORT:-8000}/api/v1/health || exit 1"

CMD ["sh", "-c", "python -m alembic upgrade head && exec uvicorn api.routes:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
