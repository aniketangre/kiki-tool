# -----------------------------------------------------------------------
# KIKI Lattice Structure Optimizer — Dockerfile
#
# Build:
#   docker build -t kiki-tool .
#
# Run:
#   docker run -p 8000:8000 kiki-tool
#
# Then open:
#   http://localhost:8000/ui    — Gradio browser form
#   http://localhost:8000/docs  — REST API docs
#
# NOTE: The trained model (src/serving/model/extra_trees_model.pkl) must
# exist locally before building. It is not tracked in git due to its size.
# Generate it by running:  python src/models/train.py
# -----------------------------------------------------------------------

# Use official Python slim image — smaller footprint than the full image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# -----------------------------------------------------------------------
# Install dependencies first (separate layer for better Docker cache reuse).
# If only source code changes, this layer is not rebuilt.
# -----------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------
# Copy application source code.
# The model artifacts (*.pkl, normalizer.csv) live inside src/serving/model/
# and are included here.
# -----------------------------------------------------------------------
COPY src/ ./src/

# Expose the port uvicorn will listen on
EXPOSE 8000

# -----------------------------------------------------------------------
# Start the API server.
# --host 0.0.0.0  makes it reachable from outside the container.
# --port 8000     matches the EXPOSE directive above.
# Remove --reload in production (it watches for file changes).
# -----------------------------------------------------------------------
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
