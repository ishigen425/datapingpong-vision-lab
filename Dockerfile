FROM ghcr.io/astral-sh/uv:0.9.17 AS uv

FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace

COPY --from=uv /uv /usr/local/bin/uv
COPY --from=uv /uvx /usr/local/bin/uvx

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv --system-site-packages /opt/venv \
    && uv pip install --python /opt/venv/bin/python -r requirements.txt

COPY src ./src
COPY tasks ./tasks

CMD ["python", "tasks/verify_torch/run.py"]
