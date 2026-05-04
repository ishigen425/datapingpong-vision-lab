from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_URL = "https://lab.osai.ai/"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attrs_dict = dict(attrs)
        self._href = attrs_dict.get("href")
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            self.links.append((self._href, "".join(self._text).strip()))
            self._href = None
            self._text = []


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and normalize OpenTTGames markup zip files.")
    parser.add_argument("--dataset-url", default=DEFAULT_DATASET_URL)
    parser.add_argument("--archive-dir", type=Path, default=ROOT / "archive/openttgames/markup")
    parser.add_argument("--extract-dir", type=Path, default=ROOT / "data/openttgames/markup")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/annotations/openttgames")
    parser.add_argument("--items", nargs="*", default=None, help="Optional item stems, e.g. game_1 test_2.")
    parser.add_argument("--skip-download", action="store_true", help="Normalize already downloaded zip files.")
    parser.add_argument("--dry-run", action="store_true", help="List resolved markup files without downloading or writing.")
    args = parser.parse_args()

    links = discover_markup_links(args.dataset_url)
    if args.items:
        wanted = set(args.items)
        links = [link for link in links if Path(urlparse(link).path).stem in wanted]

    if args.dry_run:
        print(json.dumps({"dataset_url": args.dataset_url, "markup_links": links}, indent=2))
        return 0

    args.archive_dir.mkdir(parents=True, exist_ok=True)
    args.extract_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    zip_paths = []
    for link in links:
        zip_path = args.archive_dir / Path(urlparse(link).path).name
        if not args.skip_download or not zip_path.exists():
            download(link, zip_path)
        zip_paths.append(zip_path)

    ball_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for zip_path in zip_paths:
        item = zip_path.stem
        target_dir = args.extract_dir / item
        target_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(zip_path) as archive:
            archive.extractall(target_dir)
        normalized = normalize_extracted_markup(item, target_dir)
        ball_rows.extend(normalized["ball"])
        event_rows.extend(normalized["events"])

    ball_output = args.output_dir / "ball_positions.jsonl"
    event_output = args.output_dir / "events.jsonl"
    write_jsonl(ball_output, ball_rows)
    write_jsonl(event_output, event_rows)
    print(
        json.dumps(
            {
                "downloaded_or_reused": len(zip_paths),
                "ball_rows": len(ball_rows),
                "event_rows": len(event_rows),
                "ball_output": str(ball_output),
                "event_output": str(event_output),
            },
            indent=2,
        )
    )
    return 0


def discover_markup_links(dataset_url: str) -> list[str]:
    with urlopen(dataset_url) as response:
        html = response.read().decode("utf-8")
    parser = LinkParser()
    parser.feed(html)
    links: list[str] = []
    for href, text in parser.links:
        name = Path(urlparse(href).path).name or text.split()[0]
        if _is_markup_zip_name(name):
            links.append(urljoin(dataset_url, href))
    return sorted(dict.fromkeys(links))


def download(url: str, path: Path) -> None:
    with urlopen(url) as response:
        path.write_bytes(response.read())


def normalize_extracted_markup(item: str, root: Path) -> dict[str, list[dict[str, Any]]]:
    ball_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for json_path in sorted(root.rglob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if _looks_like_ball_mapping(payload):
            ball_rows.extend(_normalize_ball_mapping(item, json_path, payload))
        elif _looks_like_event_mapping(payload):
            event_rows.extend(_normalize_event_mapping(item, json_path, payload))
        elif isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict) and "frame" in row and "event" in row:
                    event_rows.append(_event_row(item, json_path, row["frame"], row["event"], row))
                elif isinstance(row, dict) and "frame" in row and "x" in row and "y" in row:
                    ball_rows.append(_ball_row(item, json_path, row["frame"], row))
    return {"ball": ball_rows, "events": event_rows}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in sorted(rows, key=lambda item: (str(item.get("item")), int(item.get("frame", -1)), str(item.get("event", "")))):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_markup_zip_name(name: str) -> bool:
    return name.endswith(".zip") and (name.startswith("game_") or name.startswith("test_"))


def _looks_like_ball_mapping(payload: Any) -> bool:
    return isinstance(payload, dict) and bool(payload) and all(
        str(key).isdigit() and isinstance(value, dict) and "x" in value and "y" in value for key, value in payload.items()
    )


def _looks_like_event_mapping(payload: Any) -> bool:
    return isinstance(payload, dict) and bool(payload) and all(str(key).isdigit() and isinstance(value, str) for key, value in payload.items())


def _normalize_ball_mapping(item: str, path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [_ball_row(item, path, int(frame), row) for frame, row in payload.items()]


def _normalize_event_mapping(item: str, path: Path, payload: dict[str, str]) -> list[dict[str, Any]]:
    return [_event_row(item, path, int(frame), event, {}) for frame, event in payload.items()]


def _ball_row(item: str, path: Path, frame: Any, row: dict[str, Any]) -> dict[str, Any]:
    x = row.get("x")
    y = row.get("y")
    detected = x is not None and y is not None and not (float(x) == -1.0 and float(y) == -1.0)
    return {
        "source": str(path),
        "item": item,
        "frame": int(frame),
        "x": float(x) if detected else None,
        "y": float(y) if detected else None,
        "confidence": float(row.get("confidence", row.get("prod", 1.0))),
        "detected": detected,
    }


def _event_row(item: str, path: Path, frame: Any, event: Any, row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "source": str(path),
        "item": item,
        "frame": int(frame),
        "event": _normalize_event(str(event)),
    }
    for key in ("x", "y", "u", "v"):
        if key in row:
            payload[key] = row[key]
    return payload


def _normalize_event(event: str) -> str:
    normalized = event.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized == "net":
        return "net_hit"
    if normalized == "empty_event":
        return "empty"
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
