# File: /deployment/Dockerfile

# ================================
# Base Stage: System Dependencies
# ================================
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# ================================
# Dependencies Stage: Python Packages
# ================================
FROM base as dependencies

# Copy requirements file
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ================================
# API Stage: Flask Application Setup
# ================================
FROM dependencies as api-stage

# Create directories
RUN mkdir -p /app/api /app/logs /app/data

# Copy API code
COPY api/ ./api/
COPY config/ ./config/
COPY data_pipeline/ ./data_pipeline/

# Set proper permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 5000

# Health check for API service
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/health/status || exit 1

# API startup command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", \
     "--keep-alive", "5", "--max-requests", "1000", "--preload", \
     "api.app:app"]

# ================================
# Dashboard Stage: Streamlit Application Setup
# ================================
FROM dependencies as dashboard-stage

# Create directories
RUN mkdir -p /app/dashboard /app/logs /app/data

# Copy dashboard code
COPY dashboard/ ./dashboard/
COPY config/ ./config/

# Set proper permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check for dashboard service
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

# Dashboard startup command
CMD ["streamlit", "run", "dashboard/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]

# ================================
# Production Stage: Combined Services with Supervisord
# ================================
FROM dependencies as production

# Install supervisor for process management
USER root
RUN pip install supervisor

# Create application directories
RUN mkdir -p /app/api \
             /app/dashboard \
             /app/config \
             /app/data_pipeline \
             /app/logs \
             /app/data \
             /etc/supervisor/conf.d

# Copy all application code
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY config/ ./config/
COPY data_pipeline/ ./data_pipeline/

# Copy supervisor configuration
COPY deployment/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Wait for database to be ready\n\
echo "Waiting for database..."\n\
while ! nc -z ${DB_HOST:-postgres} ${DB_PORT:-5432}; do\n\
  sleep 1\n\
done\n\
echo "Database is ready!"\n\
\n\
# Initialize database if needed\n\
python -c "from api.app import init_db; init_db()"\n\
\n\
# Start supervisor\n\
exec /usr/local/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf\n' \
> /app/startup.sh && chmod +x /app/startup.sh

# Set proper permissions
RUN chown -R appuser:appuser /app /var/log

# Switch to non-root user
USER appuser

# Expose both ports
EXPOSE 5000 8501

# Combined health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:5000/api/health/status && \
        curl -f http://localhost:8501/healthz || exit 1

# Signal handling for graceful shutdown
STOPSIGNAL SIGTERM

# Production startup command
CMD ["/app/startup.sh"]
