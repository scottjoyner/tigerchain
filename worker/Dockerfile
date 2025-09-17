FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ingest_docs.py .
COPY sample_docs sample_docs
CMD ["python", "ingest_docs.py", "--help"]
