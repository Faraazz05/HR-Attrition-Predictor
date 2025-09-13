# HR Attrition Predictor - Production Dockerfile
# =============================================
# Multi-stage Docker build for optimal performance and security
# Author: HR Analytics Team
# Date: September 2025
# Version: 2.0

# ================================================================
# STAGE 1: Base Python Environment
# ================================================================

FROM python:3.11-slim as base

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH="/app"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ================================================================
# STAGE 2: Dependencies Installation  
# ================================================================

FROM base as dependencies

# Copy requirements files
COPY requirements.txt .
COPY requirements-dev.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install additional ML/Data Science packages
RUN pip install --no-cache-dir \
    streamlit==1.28.1 \
    plotly==5.17.0 \
    pandas==2.1.3 \
    numpy==1.24.3 \
    scikit-learn==1.3.2 \
    xgboost==2.0.2 \
    shap==0.43.0 \
    optuna==3.4.0 \
    psutil==5.9.6 \
    python-multipart==0.0.6 \
    pydantic==2.5.0 \
    fastapi==0.104.1 \
    uvicorn==0.24.0

# ================================================================
# STAGE 3: Application Build
# ================================================================

FROM dependencies as builder

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create necessary directories
RUN mkdir -p /app/data/processed \
    /app/data/synthetic \
    /app/logs \
    /app/models/saved \
    /app/results \
    /app/reports \
    /app/config

# Copy application code
COPY . /app/

# Copy configuration files
COPY config/ /app/config/
COPY streamlit_app/ /app/streamlit_app/
COPY src/ /app/src/

# Set proper permissions
RUN chown -R appuser:appuser /app && \
    chmod +x /app/scripts/*.sh 2>/dev/null || true

# ================================================================
# STAGE 4: Production Image
# ================================================================

FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application from builder stage
COPY --from=builder --chown=appuser:appuser /app /app

# Create streamlit config directory
RUN mkdir -p /home/appuser/.streamlit && \
    chown -R appuser:appuser /home/appuser/.streamlit

# Copy Streamlit configuration
COPY --chown=appuser:appuser <<EOF /home/appuser/.streamlit/config.toml
[server]
port = 8501
address = "0.0.0.0"
headless = true
fileWatcherType = "none"
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#00D4FF"
backgroundColor = "#0A0E27"
secondaryBackgroundColor = "#1A1F3A"
textColor = "#F0F8FF"

[client]
showErrorDetails = false
EOF

# Copy Streamlit secrets template (user should mount real secrets)
COPY --chown=appuser:appuser <<EOF /home/appuser/.streamlit/secrets.toml
# Mount your real secrets.toml file here
# Example secrets:
# [database]
# host = "localhost"
# port = 5432
# username = "user"
# password = "pass"

# [email]
# smtp_host = "smtp.gmail.com"
# smtp_port = 587
# username = "your-email@company.com"
# password = "your-app-password"
EOF

# Create startup script
COPY --chown=appuser:appuser <<'EOF' /app/start.sh
#!/bin/bash
set -e

echo "🚀 Starting HR Attrition Predictor..."

# Create necessary directories
mkdir -p /app/data/processed /app/data/synthetic /app/logs /app/results

# Initialize synthetic data if not exists
if [ ! -f "/app/data/synthetic/hr_employees.csv" ]; then
    echo "📊 Generating synthetic data..."
    python -c "
import sys
sys.path.append('/app')
from src.data_processing.data_loader import generate_synthetic_hr_data
generate_synthetic_hr_data()
print('✅ Synthetic data generated')
"
fi

# Start the application
echo "🌟 Launching Streamlit application..."
exec streamlit run streamlit_app/Dashboard.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.fileWatcherType=none
EOF

RUN chmod +x /app/start.sh

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/start.sh"]

# ================================================================
# METADATA
# ================================================================

LABEL maintainer="HR Analytics Team" \
      version="2.0" \
      description="HR Attrition Predictor - AI-powered employee retention analytics" \
      org.opencontainers.image.title="HR Attrition Predictor" \
      org.opencontainers.image.description="Advanced ML system for predicting employee attrition with cyberpunk UI" \
      org.opencontainers.image.version="2.0" \
      org.opencontainers.image.created="2025-09-13" \
      org.opencontainers.image.source="https://github.com/your-org/hr-attrition-predictor" \
      org.opencontainers.image.licenses="MIT"

# ================================================================
# BUILD ARGS AND RUNTIME INFO
# ================================================================

# Build arguments for customization
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Runtime labels
LABEL build_date=$BUILD_DATE \
      vcs_ref=$VCS_REF \
      version=$VERSION

# Default command (can be overridden)
CMD ["/app/start.sh"]
