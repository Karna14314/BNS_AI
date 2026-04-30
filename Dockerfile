FROM python:3.11-slim

# Install system dependencies (cmake is required to build llama-cpp)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (UID 1000 is required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies
# We use CMAKE_ARGS to ensure a clean CPU-only build during the Docker layer creation
COPY --chown=user requirements.txt .
RUN CMAKE_ARGS="-DGGML_BLAS=OFF -DGGML_METAL=OFF" \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user . .

# Set environment variables for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
