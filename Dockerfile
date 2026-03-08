FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

RUN pip install --no-cache-dir \
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

CMD ["python3", "train.py"]
