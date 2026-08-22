from __future__ import annotations

from rag.config import collection_cfg, load_config, resolve


def test_paths_are_relative_to_the_rag_folder(tmp_path):
    workdir = tmp_path / "project"
    rag_dir = workdir / "RAG"
    docs_dir = workdir / "docs"
    rag_dir.mkdir(parents=True)
    docs_dir.mkdir()
    config_path = rag_dir / "config.yaml"
    config_path.write_text(
        "docs_path: ../docs\n"
        "store_dir: indexes\n"
        "collections:\n"
        "  default:\n"
        "    path: ../docs\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    collection = collection_cfg(cfg, "default")

    assert resolve(cfg, collection["path"]) == docs_dir.resolve()
    assert resolve(cfg, cfg["store_dir"]) == (rag_dir / "indexes").resolve()


def test_collection_defaults_match_the_portable_design(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    collection = collection_cfg(load_config(config_path), "notes")

    assert collection == {
        "name": "notes",
        "path": "../docs",
        "chunk": "parent_child",
        "child_tokens": 256,
        "parent_tokens": 1024,
        "overlap_tokens": 32,
    }
