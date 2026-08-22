# RAG drop-in local

Ce dossier est un kit autonome à copier sous `<workdir>/RAG/`, à côté de `<workdir>/docs/`. Il n’utilise que des services locaux : Ollama sur `http://127.0.0.1:11434` et un fichier SQLite par collection sous `RAG/indexes/`.

## Installation

Depuis `<workdir>` :

```bash
python3 -m venv RAG/.venv
RAG/.venv/bin/pip install -r RAG/requirements.txt
ollama pull embeddinggemma
ollama pull qwen3.5:0.8b
```

Les scripts redémarrent automatiquement avec `RAG/.venv/bin/python` lorsque le venv existe.

## Ingestion et recherche

```bash
python RAG/ingest.py
python RAG/ingest.py --path docs --collection default
python RAG/ingest.py --config RAG/config.yaml

python RAG/search.py "question"
python RAG/search.py --query "question" --k 8
python RAG/search.py --query "question" --no-rewrite
python RAG/search.py --query "question" --collection default
```

Chaque commande écrit un seul objet JSON sur stdout. Les diagnostics éventuels vont sur stderr. L’ingestion renvoie notamment `files`, `chunks`, `stats` et le chemin absolu de l’index. Une recherche réussie suit ce contrat :

```json
{
  "ok": true,
  "query": "question",
  "rewritten": ["question", "reformulation"],
  "hyde": "paragraphe hypothétique",
  "collection": "default",
  "hits": [
    {
      "text": "contexte parent",
      "score": 1.0,
      "path": "../docs/note.md",
      "page": null,
      "locator": "Titre de section",
      "collection": "default"
    }
  ]
}
```

Codes de sortie : `0` si la recherche trouve des résultats, `1` si l’index valide ne produit aucun résultat, `2` pour une requête manquante, un index vide ou une erreur. `--no-rewrite` conserve uniquement la question originale. Par défaut, `qwen3.5:0.8b` produit deux ou trois reformulations et un texte HyDE ; si cette génération ou son JSON échoue, la question originale reste utilisée.

## Architecture et configuration

- `embeddinggemma` est l’unique modèle dense, à l’ingestion comme à la requête.
- SQLite FTS5 fournit BM25 ; la fusion dense/lexicale utilise RRF.
- `parent_child` est la stratégie par défaut : enfants d’environ 256 tokens indexés, parents d’environ 1024 tokens renvoyés.
- `section` indexe et renvoie chaque section avec son titre ou son libellé `Article …`.
- `window` utilise des fenêtres glissantes et renvoie `locator: null`.
- Les hits exposent uniquement `path`, `page`, `locator` et `collection` comme métadonnées.

Les chemins du YAML sont relatifs au dossier contenant `config.yaml`. Avec `RAG/config.yaml`, `../docs` vise donc `<workdir>/docs`, tandis que `indexes` vise `<workdir>/RAG/indexes`. Un chemin donné par `--path` suit les règles usuelles du shell et est résolu depuis le répertoire courant.

Le kit ne contient ni serveur HTTP, ni orchestration, ni types métier. Docker, les services distants, FastAPI, Qdrant, LangChain, torch, le rerank, ColBERT et SPLADE sont hors périmètre.

## Tests

```bash
RAG/.venv/bin/python -m pytest RAG/tests -q
RAG/.venv/bin/python -m pytest RAG/tests -q -m ollama
```

Le smoke test est ignoré automatiquement si Ollama ou `embeddinggemma` n’est pas disponible localement.

---

English: copy this folder to `<workdir>/RAG/`, install `requirements.txt`, and invoke `ingest.py` or `search.py` from `<workdir>`. Both CLIs emit one JSON object on stdout. All retrieval stays local.
