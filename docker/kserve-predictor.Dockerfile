FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY jobs /app/jobs
COPY output/serving /app/output/serving

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

EXPOSE 8080

CMD ["uvicorn", "jobs.kserve.predictor:app", "--host", "0.0.0.0", "--port", "8080"]
