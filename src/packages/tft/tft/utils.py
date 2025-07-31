from __future__ import annotations
from pathlib import Path
import json


def ensure_dir(path: Path) -> None:
    """
    Ensure the given directory exists.
    Creates all parent folders if necessary (no-op if already exists).
    """
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    """
    Serialize `obj` as JSON (indented) and write to `path`.
    Automatically creates parent directory via ensure_dir.
    """
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))
