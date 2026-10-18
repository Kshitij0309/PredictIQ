# ==============================================================================
# PredictIQ — Production REST API Service Dockerfile
# ==============================================================================
# Multi-layer, reproducible container running FastAPI with non-root security.
# Base Image: Python 3.13-slim (Debian Bookworm)
# ==============================================================================

FROM python:3.13-slim AS runtime

# Set non-interactive debconf & Python environment flags
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENVIRONMENT=production \
    PREDICTIQ_ROOT_DIR=/app \
    MODEL_PATH=/app/models/churn_champion_model.joblib \
    PORT=8000 \
    HOST=0.0.0.0

# Install minimal OS dependencies for health checks & library compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Set working directory
WORKDIR /app

# Copy packaging configuration and source tree
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install application package with api dependencies
RUN pip install --no-cache-dir ".[api]"

# Copy champion model artifacts
COPY models/churn_champion_model.joblib models/churn_champion_model_metadata.json ./models/

# Ensure application files are owned by the non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root execution
USER appuser

# Expose HTTP service port
EXPOSE 8000

# Container liveness health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production Uvicorn ASGI server
CMD ["uvicorn", "predictiq.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
