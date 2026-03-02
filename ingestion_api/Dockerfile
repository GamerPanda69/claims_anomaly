FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Copy the shared folder and the service code
#COPY ./shared /app/shared
#COPY ./ingestion_api /app/ingestion_api

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
