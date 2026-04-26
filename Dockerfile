# Use a lightweight Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Tell OpenEnv server which port to use
ENV OPENENV_PORT=8000
ENV OPENENV_HOST=0.0.0.0

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Port 7860 — Flask UI (HuggingFace health-check target)
EXPOSE 7860
# Port 8000 — OpenEnv HTTP + WebSocket server (EnvClient target)
EXPOSE 8000

# Run both services via start.sh.
# Flask (7860) is the foreground process so HF sees the Space as healthy.
# OpenEnv server (8000) runs in the background alongside it.
CMD ["bash", "start.sh"]
