# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose HTTPS port
EXPOSE 8008

# Run Uvicorn with SSL
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8008", "--ssl-keyfile", "key.pem", "--ssl-certfile", "cert.pem"]
