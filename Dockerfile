FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --break-system-packages --upgrade pip \
    && python -m pip install --break-system-packages -r requirements.txt

COPY src ./src
COPY tasks ./tasks

CMD ["python", "tasks/verify_torch/run.py"]
