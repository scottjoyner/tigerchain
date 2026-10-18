FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app/worker
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY worker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY worker/ ./
COPY gsql/ /app/gsql/
RUN python -m compileall -q /app/worker
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
