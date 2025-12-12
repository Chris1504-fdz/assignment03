# Dockerfile for Assignment 3 (RLHF: Reward, PPO, GRPO, DPO, Eval)
# CPU-first and grader-friendly. You can still run it on a GPU machine (it will use CUDA if your torch install supports it).

FROM python:3.10-slim

WORKDIR /workspace

# Basic quality-of-life + Hugging Face caches inside the container (so outputs are reproducible and self-contained)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

# System deps (keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better Docker layer caching)
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt

# Copy project code (models/ and eval_outputs/ should be excluded via .dockerignore)
COPY . /workspace

# Default action when you run the container with no extra arguments:
# It will run the evaluation script and write artifacts under eval_outputs/<timestamp>/.
CMD ["python", "src/eval_all_policies.py"]