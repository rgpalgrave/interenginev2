# Use official Python runtime as base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=True \
    APP_HOME=/app \
    PORT=8080 \
    PYTHONPATH=/app:$PYTHONPATH

# Set working directory
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY integrated_streamlit_app.py .
COPY interstitial_engine.py .
COPY position_calculator.py .

# Verify files were copied
RUN echo "=== Files in container ===" && \
    ls -la /app/*.py && \
    echo "" && \
    echo "=== Testing imports ===" && \
    python3 -c "import sys; print('Python path:', sys.path[:3]); from interstitial_engine import LatticeParams; print('✓ interstitial_engine imported successfully')" && \
    python3 -c "from position_calculator import generate_metal_positions; print('✓ position_calculator imported successfully')" && \
    echo "=== Import test passed ===" || \
    (echo "✗ Import failed - this will cause the app to fail"; exit 1)

# Create directory for Streamlit config
RUN mkdir -p ~/.streamlit

# Create Streamlit config file for Cloud Run
RUN printf '[general]\nheadless = true\n\n[server]\nport = 8080\nenableCORS = false\nenableXsrfProtection = false\n\n[logger]\nlevel = "info"\n\n[theme]\nprimaryColor = "#1f77b4"\nbackgroundColor = "#ffffff"\nsecondaryBackgroundColor = "#f0f2f6"\ntextColor = "#262730"\n' > ~/.streamlit/config.toml

# Expose port
EXPOSE 8080

# Run Streamlit
CMD ["streamlit", "run", "integrated_streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
