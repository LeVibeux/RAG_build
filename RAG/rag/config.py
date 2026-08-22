"""Shared paths and YAML config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

RAG_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = RAG_ROOT / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = (path or DEFAULT_CONFIG).expanduser().resolve()
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a YAML mapping: {cfg_path}")
    data["_config_path"] = cfg_path
    # Config-relative paths make the kit portable: with <workdir>/RAG/config.yaml,
    # ../docs is <workdir>/docs while indexes is <workdir>/RAG/indexes.
    data["_root"] = cfg_path.parent
    return data


def resolve(cfg: dict[str, Any], rel: str | Path) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return (cfg["_root"] / p).resolve()


def collection_cfg(cfg: dict[str, Any], name: str = "default") -> dict[str, Any]:
    cols = cfg.get("collections") or {}
    raw = dict(cols.get(name) or {})
    merged = {
        "name": name,
        "path": raw.get("path", cfg.get("docs_path", "../docs")),
        "chunk": raw.get("chunk", cfg.get("chunk", "parent_child")),
        "child_tokens": int(raw.get("child_tokens", cfg.get("child_tokens", 256))),
        "parent_tokens": int(raw.get("parent_tokens", cfg.get("parent_tokens", 1024))),
        "overlap_tokens": int(raw.get("overlap_tokens", cfg.get("overlap_tokens", 32))),
    }
    return merged
