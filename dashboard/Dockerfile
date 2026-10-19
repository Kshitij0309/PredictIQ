# ==============================================================================
# PredictIQ — Streamlit Dashboard Dockerfile
# ==============================================================================
# Multi-layer, reproducible container running Streamlit with non-root security.
# Base Image: Python 3.13-slim (Debian Bookworm)
# ==============================================================================

FROM python:3.13-slim AS runtime

# Set non-interactive debconf & Python environment flags
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PREDICTIQ_ROOT_DIR=/app \
    PREDICTIQ_API_URL=https://predictiq-u912.onrender.com \
    PREDICTIQ_DATA_DIR=/app/data \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install minimal OS dependencies for health checks & library compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Set working directory
WORKDIR /app

# Copy dashboard requirements and install dependencies
COPY dashboard/requirements.txt ./dashboard/requirements.txt
RUN pip install --no-cache-dir -r ./dashboard/requirements.txt

# Copy dashboard application tree and Streamlit configuration
COPY dashboard/ ./dashboard/
COPY .streamlit/ ./.streamlit/

# Copy dataset, evaluation reports, and model metadata for analytics & intelligence
COPY data/ ./data/
COPY reports/ ./reports/
COPY models/churn_champion_model_metadata.json ./models/

# Ensure application files are owned by the non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root execution
USER appuser

# Expose Streamlit default port (default 8501, dynamically overridden by $PORT on platforms like Render)
EXPOSE 8501

# Container health check probe targeting Streamlit internal health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Launch production Streamlit application from /app with dynamic $PORT support
CMD ["sh", "-c", "exec python -m streamlit run dashboard/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --browser.gatherUsageStats=false"]
