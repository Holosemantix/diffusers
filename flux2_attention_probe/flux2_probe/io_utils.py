from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_yaml(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    if path is None:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
    return data


def save_yaml(data: dict[str, Any], path: str | os.PathLike[str]) -> None:
    import yaml

    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def save_json(data: Any, path: str | os.PathLike[str], indent: int = 2) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=json_default)
        f.write("\n")


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(record: dict[str, Any], path: str | os.PathLike[str]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")


def json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def copy_if_exists(src: str | os.PathLike[str] | None, dst: str | os.PathLike[str]) -> None:
    if src is None:
        return
    src_path = Path(src)
    if src_path.exists():
        ensure_dir(Path(dst).parent)
        shutil.copy2(src_path, dst)


def add_local_diffusers_to_path(path: str | os.PathLike[str] | None = None) -> Path | None:
    """Add a source checkout's ``src`` directory to sys.path when present."""

    candidates = []
    if path:
        candidates.append(Path(path))
    env_path = os.environ.get("DIFFUSERS_SRC")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            project_root().parent / "diffusers" / "src",
            Path.cwd() / "diffusers" / "src",
            Path.cwd().parent / "diffusers" / "src",
        ]
    )
    for candidate in candidates:
        src = candidate
        if (src / "src" / "diffusers").is_dir():
            src = src / "src"
        if (src / "diffusers").is_dir():
            src_str = str(src.resolve())
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            return src
    return None


class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

