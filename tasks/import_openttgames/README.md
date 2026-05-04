# import_openttgames

Downloads OpenTTGames markup zip files from `https://lab.osai.ai/`, extracts them, and writes normalized JSONL annotations.

```bash
python3 tasks/import_openttgames/run.py
```

Useful options:

```bash
python3 tasks/import_openttgames/run.py --dry-run
python3 tasks/import_openttgames/run.py --items game_1 test_1
python3 tasks/import_openttgames/run.py --skip-download
```

Outputs:

- `archive/openttgames/markup/*.zip`
- `data/openttgames/markup/*`
- `data/annotations/openttgames/ball_positions.jsonl`
- `data/annotations/openttgames/events.jsonl`

OpenTTGames is CC BY-NC-SA 4.0.
