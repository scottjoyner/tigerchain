FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ingest_docs.py .
COPY sample_docs sample_docs
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
