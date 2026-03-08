FROM vllm/vllm-openai:latest

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

COPY HFToversight/ /app

ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

RUN python3 -c "import vllm; print('vllm OK')"
RUN python3 -c "from trl import GRPOConfig, GRPOTrainer; print('TRL OK')"
RUN python3 smoke_test.py

CMD ["sleep", "infinity"]
