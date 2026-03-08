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

# Install Python deps
RUN uv pip install --system \
    torch==2.5.1 \
    transformers>=4.46.0 \
    trl>=0.12.0 \
    peft>=0.13.0 \
    accelerate>=1.0.0 \
    vllm>=0.6.0 \
    datasets>=3.0.0 \
    huggingface_hub>=0.26.0 \
    pydantic>=2.0.0 \
    openenv-core>=0.2.0 \
    matplotlib>=3.8.0 \
    flash-attn --no-build-isolation

# Copy project
COPY HFToversight/ /app

ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

CMD ["python3", "train.py"]
