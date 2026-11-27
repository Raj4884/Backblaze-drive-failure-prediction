FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DRIVE_FAILURE_MODEL=/app/models/model.joblib

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[xgb]"

# Train outside the image (see README) and mount the model directory.
VOLUME ["/app/models"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "drive_failure.api:app", "--host", "0.0.0.0", "--port", "8000"]
