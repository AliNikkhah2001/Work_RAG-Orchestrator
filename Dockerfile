# syntax=docker/dockerfile:1
FROM python:3.11-slim
WORKDIR /app
ARG PIP_INDEX_URL=
ARG PIP_EXTRA_INDEX_URL=
ARG PIP_TRUSTED_HOST=
ENV PIP_DEFAULT_TIMEOUT=120
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
# Keep src/ layout so hatch force-include paths resolve (fallback_packs).
COPY src ./src
COPY scripts ./scripts
# BuildKit cache mount reuses downloaded wheels across builds on this host.
# Optional PIP_* build-args (Iranian mirrors) — only export when non-empty.
RUN --mount=type=cache,target=/root/.cache/pip \
    set -e; \
    if [ -n "$PIP_INDEX_URL" ]; then export PIP_INDEX_URL; fi; \
    if [ -n "$PIP_EXTRA_INDEX_URL" ]; then export PIP_EXTRA_INDEX_URL; fi; \
    if [ -n "$PIP_TRUSTED_HOST" ]; then export PIP_TRUSTED_HOST; fi; \
    pip install --default-timeout=120 -e .
ENV PYTHONPATH=/app/src:/app/scripts
EXPOSE 8100
CMD ["python", "-m", "work_rag_orchestrator.api"]
