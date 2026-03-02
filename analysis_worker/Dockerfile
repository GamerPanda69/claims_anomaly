FROM python:3.10-slim

WORKDIR /app

# Install PyTorch CPU-only first (avoids pulling the huge CUDA build from PyPI)
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py /app/

# Models will be mounted as volume at /app/models
ENV MODELS_DIR=/app/models

# Set Python to run in unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

CMD ["python", "worker.py"]
