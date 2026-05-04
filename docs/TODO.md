# TODO

## Docker Python Package Management

Resolved in the Dockerfile by copying the `uv` binary from `ghcr.io/astral-sh/uv`, creating `/opt/venv` with `--system-site-packages` so the official PyTorch image packages remain visible, and installing runtime dependencies with:

```bash
uv pip install --python /opt/venv/bin/python -r requirements.txt
```

Follow-up options:

- Move from `requirements.txt` to `pyproject.toml`
- Add `uv.lock` for fully reproducible dependency resolution
- Decide whether to move the Docker dependency source from `requirements.txt` to a project-level dependency table
