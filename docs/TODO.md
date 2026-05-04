# TODO

## Docker Python Package Management

Current Docker setup installs Python packages with:

```bash
python -m pip install --break-system-packages -r requirements.txt
```

This was added because the current PyTorch base image marks the Python environment as externally managed.

Consider switching Docker package management to `uv`:

- Avoid relying on `--break-system-packages`
- Make dependency resolution and installation faster
- Prepare for lockfile-based reproducible environments
- Decide whether to keep `requirements.txt` or move to `pyproject.toml` + `uv.lock`

Candidate direction:

```dockerfile
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
```

Need to verify interaction with the official `pytorch/pytorch` Docker image before changing the default.
