FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose ports
EXPOSE 5000 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=api/rest_server.py

# Health check (works for both Flask:5000 and Streamlit:8501)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8501/_stcore/health'); r.raise_for_status()" || \
    python -c "import requests; requests.get('http://localhost:5000/api/health')"

# Default command (API server)
CMD ["python", "api/rest_server.py"]
