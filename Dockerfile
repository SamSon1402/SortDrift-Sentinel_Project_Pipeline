FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY dashboard ./dashboard
RUN pip install --no-cache-dir .

ENV SORTDRIFT_LOCAL_DIR=/data
VOLUME ["/data"]
EXPOSE 8010
CMD ["uvicorn", "sort_drift_sentinel.api.main:app", "--host", "0.0.0.0", "--port", "8010"]
