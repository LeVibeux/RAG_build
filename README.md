# RAG portable local

Kit de recherche documentaire copiable, entièrement local. Il combine les embeddings Ollama `embeddinggemma` et BM25 via SQLite FTS5, puis fusionne les résultats avec RRF.

```text
<workdir>/
  docs/          # .md, .txt, .rst, .org, .pdf
  RAG/           # le kit et ses index SQLite
```

Depuis `<workdir>` :

```bash
python3 -m venv RAG/.venv
RAG/.venv/bin/pip install -r RAG/requirements.txt
ollama pull embeddinggemma
ollama pull qwen3.5:0.8b

python RAG/ingest.py
python RAG/search.py "votre question"
python RAG/search.py --query "votre question" --k 8 --no-rewrite
```

Les deux scripts écrivent exactement un objet JSON sur stdout. Voir [`RAG/README.md`](RAG/README.md) pour le contrat, la configuration et les tests.

Le périmètre exclut Docker, les API/services distants, FastAPI, les agents, Qdrant, LangChain, torch, le rerank, ColBERT et SPLADE.

---

Portable, local-only document retrieval kit. Copy `RAG/` next to a `docs/` folder, install its requirements, then run `python RAG/ingest.py` and `python RAG/search.py "question"` from the parent work directory.
