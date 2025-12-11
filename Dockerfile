# Multi-stage Dockerfile for E-Commerce Fraud Detection API
# Optimized for production deployment with minimal image size

# Stage 1: Builder
FROM python:3.12-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies and uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv
RUN uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/models && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser predict.py .
COPY --chown=appuser:appuser train.py .
COPY --chown=appuser:appuser src/ ./src/

# Copy model artifacts
# Note: These must exist before building the image
# Run train.py locally first to generate model files
COPY --chown=appuser:appuser models/*.json models/
COPY --chown=appuser:appuser models/*.joblib models/

# Switch to non-root user
USER appuser

# Expose port (Cloud Run will set PORT env var, defaults to 8080)
EXPOSE 8080

# Health check - uses PORT environment variable
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os; import requests; port = os.getenv('PORT', '8000'); requests.get(f'http://localhost:{port}/health').raise_for_status()" || exit 1

# Run the FastAPI application using predict.py (reads PORT from environment)
CMD ["python", "predict.py"]
