# syntax=docker/dockerfile:1.4
FROM python:3.10-slim

# System dependencies for torch, sentence-transformers, and others
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Install Python dependencies in two steps for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# If you use sentence-transformers or torch directly, ensure they're in requirements.txt
# Otherwise, you can add them here:
# RUN pip install --no-cache-dir torch sentence-transformers

# Copy the rest of the code
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Default command: run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]