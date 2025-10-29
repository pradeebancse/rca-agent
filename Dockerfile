# -------------------------------
# Stage 1: Base image with Python
# -------------------------------
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# -------------------------------
# Stage 2: Run app
# -------------------------------
# ✅ Use correct Python module path - file is in current directory
CMD ["uvicorn", "log_analysis_agent:app", "--host", "0.0.0.0", "--port", "8000"]
