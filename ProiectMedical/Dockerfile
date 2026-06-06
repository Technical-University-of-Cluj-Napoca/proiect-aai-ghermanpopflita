# Use official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8501

# Set work directory
WORKDIR /app

# Install system dependencies needed for compiling packages if any
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ /app/src/
COPY corpus/ /app/corpus/
COPY scripts/ /app/scripts/
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

# Create folders for runtime logs and data
RUN mkdir -p /app/logs /app/data /app/vectorstore

# Make entrypoint executable
RUN chmod +x /app/docker-entrypoint.sh

# Expose Streamlit port
EXPOSE 8501

# Healthcheck to verify the web server is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Entrypoint auto-builds vectorstore if missing, then starts Streamlit
CMD ["/app/docker-entrypoint.sh"]
