FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src/work_rag_orchestrator ./work_rag_orchestrator
COPY scripts ./scripts
RUN pip install --no-cache-dir -e .
ENV PYTHONPATH=/app:/app/scripts
EXPOSE 8100
CMD ["python", "-m", "work_rag_orchestrator.api"]
