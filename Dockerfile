FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip git curl && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Install uv for fast dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

# Install in stages to avoid resolution conflicts
# 1. vllm first (pickiest about torch version)
RUN uv pip install --system vllm

# 2. Training stack (let it resolve against installed torch)
RUN uv pip install --system \
    trl \
    peft \
    accelerate \
    datasets \
    huggingface_hub \
    pydantic \
    openenv-core \
    matplotlib

# Copy project
COPY HFToversight/ /app

ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Verify imports at build time
RUN python3 -c "from trl import GRPOConfig, GRPOTrainer; print('TRL OK')"
RUN python3 smoke_test.py

# Install flash-attn at runtime (needs GPU), then run training
CMD pip install flash-attn --no-build-isolation && python3 train.py
