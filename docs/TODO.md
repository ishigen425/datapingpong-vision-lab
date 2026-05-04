# TODO

## Docker Python Package Management

Resolved in the Dockerfile by copying the `uv` binary from `ghcr.io/astral-sh/uv`, creating `/opt/venv` with `--system-site-packages` so the official PyTorch image packages remain visible, and syncing runtime dependencies from `pyproject.toml` with `uv.lock`:

```bash
uv sync --locked --active --inexact
```
