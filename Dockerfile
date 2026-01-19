# Dockerfile for Oracle Lottery Predictor
# Multi-stage build for optimized image size

# Build stage
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies required for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies for Qt and OpenGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libgl1 \
    libopengl0 \
    libglx0 \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb1 \
    libx11-xcb1 \
    libx11-6 \
    libdbus-1-3 \
    libxi6 \
    libxcursor1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY src/ /app/src/
COPY pyproject.toml /app/
COPY README.md LICENSE.txt /app/
COPY data/ /app/data/
COPY assets/ /app/assets/

# Create necessary directories
RUN mkdir -p /app/exports /app/data/history

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen \
    PYTHONPATH=/app

# Expose port if needed for future web interface
EXPOSE 8080

# Set default command to show help
CMD ["python", "src/ulh_desktop.py", "--help"]
