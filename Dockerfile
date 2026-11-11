# syntax=docker/dockerfile:1
FROM python:3.11-slim
WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=120
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
# Keep src/ layout so hatch force-include paths resolve (fallback_packs).
COPY src ./src
COPY scripts ./scripts
# BuildKit cache mount reuses downloaded wheels across builds on this host.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=120 -e .
ENV PYTHONPATH=/app/src:/app/scripts
EXPOSE 8100
CMD ["python", "-m", "work_rag_orchestrator.api"]
