# Subtitle Translator Dockerfile
FROM python:3.14-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and project files
COPY pyproject.toml .
COPY src/ ./src/

# Install the package itself (makes subtitle_translator importable)
RUN pip install --no-cache-dir -e .

# Create data directory for persistence and non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app
USER appuser

# Persistent data volume (encryption keys, job database)
VOLUME ["/app/data"]

# Expose port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

# Run the application
CMD ["uvicorn", "subtitle_translator.main:app", "--host", "0.0.0.0", "--port", "8765"]