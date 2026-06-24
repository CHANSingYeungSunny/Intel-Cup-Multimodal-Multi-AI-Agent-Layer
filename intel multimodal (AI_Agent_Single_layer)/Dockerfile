# Single AI Agent Layer — Multi-stage Dockerfile
#
# Build:  docker build -t ai-agent-layer .
# Run:    docker run -p 8000:8000 ai-agent-layer

# ---- Stage 1: Builder ------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: Runtime ------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY config.py .
COPY database.py .
COPY models.py .
COPY schemas.py .
COPY decision_engine.py .
COPY trend_analyzer.py .
COPY advice_generator.py .
COPY main.py .

# Create a non-root user
RUN groupadd -r agent && useradd -r -g agent agent \
    && chown -R agent:agent /app
USER agent

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["uvicorn", "main:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
