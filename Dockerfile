# Multi-stage Docker build for AAS LanceDB MCP Server
FROM python:3.13-slim as builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies in virtual environment
RUN uv sync --frozen --no-install-project

# Production stage
FROM python:3.13-slim as production

# Install system dependencies needed for sentence transformers
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy uv from builder
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder /usr/local/bin/uvx /usr/local/bin/uvx

# Create non-root user
RUN groupadd --gid 1001 mcp \
    && useradd --uid 1001 --gid mcp --shell /bin/bash --create-home mcp

# Set working directory and ownership
WORKDIR /app
RUN chown -R mcp:mcp /app

# Switch to non-root user
USER mcp

# Copy virtual environment from builder
COPY --from=builder --chown=mcp:mcp /app/.venv /app/.venv

# Copy application code
COPY --chown=mcp:mcp . .

# Install the application
RUN uv pip install --no-deps .

# Create directory for LanceDB data
RUN mkdir -p /app/data && chmod 755 /app/data

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    LANCEDB_URI="/app/data/.aas_lancedb" \
    EMBEDDING_MODEL="all-MiniLM-L6-v2" \
    EMBEDDING_DEVICE="cpu"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from lancedb_mcp import server; print('healthy')" || exit 1

# Expose MCP server (stdio by default, but can be configured for HTTP)
EXPOSE 8000

# Default command
ENTRYPOINT ["python", "-m", "lancedb_mcp"]
CMD ["--help"]

# Labels for better metadata
LABEL org.opencontainers.image.title="AAS LanceDB MCP Server" \
      org.opencontainers.image.description="Enhanced LanceDB MCP server for arbitrary datastore management with sentence transformers" \
      org.opencontainers.image.vendor="Applied AI Systems" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/applied-ai-systems/aas-lancedb-mcp" \
      org.opencontainers.image.documentation="https://github.com/applied-ai-systems/aas-lancedb-mcp/blob/main/README.md"