FROM python:3.13-slim

WORKDIR /app

# Install system dependencies if any package requires it
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy required code and model assets
COPY api/ ./api
COPY models/ ./models
COPY src/ ./src

# Expose FastAPI port
EXPOSE 8000

# Run uvicorn server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
