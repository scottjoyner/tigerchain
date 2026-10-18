FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app/worker
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY worker/requirements.txt ./requirements.txt
# The API is a CPU-only retrieval service; inference GPUs are managed by Fleet.
# Install the CPU wheel first so sentence-transformers cannot pull CUDA runtimes.
ARG TORCH_CPU_VERSION=2.5.1+cpu
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_CPU_VERSION}" \
    && pip install --no-cache-dir -r requirements.txt \
    && python -c "import torch; assert torch.version.cuda is None, 'CUDA torch is not permitted in the CPU API image'"
COPY worker/ ./
COPY gsql/ /app/gsql/
RUN python -m compileall -q /app/worker
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
