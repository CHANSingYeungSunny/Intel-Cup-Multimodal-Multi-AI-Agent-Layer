# =============================================================================
# Multi-AI Agent Layer — Dockerfile
#
# Multi-stage build for the FastAPI multi-agent service.
# The Single AI Agent Layer is copied alongside as a dependency.
# =============================================================================

FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Create non-root user
RUN groupadd -r agent && useradd -r -g agent agent

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy Single AI Agent Layer (read-only dependency)
COPY "intel multimodal (AI_Agent_Single_layer)" /app/single_layer/

# Copy Multi AI Agent Layer source
COPY __init__.py .
COPY config.py .
COPY database.py .
COPY models.py .
COPY schemas.py .
COPY main.py .
COPY mcp_server.py .
COPY agent_coordinator.py .
COPY skills/ ./skills/

# Set Python path so Multi can import from Single
ENV PYTHONPATH="/app:/app/single_layer"

# Runtime configuration
ENV APP_TITLE="Multi AI Agent Layer"
ENV APP_VERSION="2.0.0"
ENV LLM_BACKEND="none"

# Use non-root user
USER agent

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/v1/health').raise_for_status()" || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:create_multi_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
