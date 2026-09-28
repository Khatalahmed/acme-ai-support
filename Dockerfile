# ACME support agent - production image.
# Build:  docker build -t acme-support .
# Run:    docker run --rm -p 8000:8000 --env-file .env acme-support   (keys come from the env,
#         never from the image - see .dockerignore)
FROM python:3.13-slim

# uv pinned to the version the project uses; dependencies installed exactly as in uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Non-root user: if the app were ever compromised, it isn't root in the container.
RUN useradd --create-home --uid 10001 app && chown app /app
USER app

# Dependencies first, so code changes don't re-install them (Docker layer caching)
COPY --chown=app pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

COPY --chown=app src ./src
COPY --chown=app data/policies ./data/policies

# Build the RAG index into the image (as the app user, so the embedding model is cached in
# its home directory) - the container starts ready, with no download on the first request.
RUN uv run --no-sync python src/rag/build_index.py

ENV PATH="/app/.venv/bin:$PATH" \
    LLM_BACKEND=azure \
    ROUTER_BACKEND=jev \
    LANGFUSE_TRACING_ENABLED=false

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
