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
python RAG/ingest.py --force          # re-découpe tout, même les fichiers inchangés

python RAG/search.py "question"
python RAG/search.py --query "question" --k 8
python RAG/search.py --query "question" --no-rewrite
python RAG/search.py --query "question" --collection default
python RAG/search.py --list-collections
```

Chaque commande écrit un seul objet JSON sur stdout. Les diagnostics éventuels vont sur stderr.

### Ingestion incrémentale

L’ingestion est incrémentale ; la relancer après chaque ajout de documents est peu coûteux :

1. un fichier dont la date de modification, la taille **et** les réglages (modèle d’embedding, stratégie et tailles de chunk) sont identiques au dernier passage est ignoré sans même être relu ;
2. sinon son sha256 est recalculé : contenu identique → ignoré aussi ;
3. sinon le fichier est re-découpé, mais chaque chunk dont le texte a déjà été embarqué (dans n’importe quel document de la collection) réutilise son vecteur. Modifier une section d’un long document ne ré-embarque que cette section ;
4. les documents disparus du dossier source sont retirés de l’index.

Changer `embed`, `chunk`, `child_tokens`, `parent_tokens` ou `overlap_tokens` force automatiquement la ré-ingestion des fichiers concernés. `--force` sert uniquement si un fichier a été modifié en conservant sa date et sa taille (ex. `rsync -t`).

La sortie d’ingestion contient : `files` (fichiers (ré)indexés), `skipped` (inchangés), `removed` (retirés de l’index), `chunks` (chunks écrits), `embedded` (textes envoyés à Ollama), `reused` (vecteurs réutilisés), `stats` et le chemin absolu de l’index.

`--list-collections` renvoie, pour chaque collection configurée ou présente dans `indexes/` : `name`, `configured`, `docs_path`, `index`, `indexed`, `documents`, `chunks`.

### Recherche

Une recherche réussie suit ce contrat :

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
- SQLite FTS5 fournit BM25 (termes combinés en `OR`, l’IDF de BM25 pénalise les mots fréquents) ; la fusion dense/lexicale utilise RRF, pondérable via `dense_weight`, `bm25_weight` et `rrf_k` dans `config.yaml`. Monter `bm25_weight` (ex. `1.5`) aide sur un corpus riche en termes exacts : noms de gènes, acronymes, identifiants.
- `embed_batch` règle le nombre de textes par requête `/api/embed` ; `embed_workers > 1` envoie plusieurs lots en parallèle, utile seulement si Ollama est lancé avec `OLLAMA_NUM_PARALLEL > 1`.
- La recherche ne charge que les identifiants et vecteurs de la collection, puis les textes des seuls hits retenus. Au-delà de quelques centaines de milliers de chunks, la recherche dense exhaustive (produit matriciel numpy) devient le facteur limitant.
- Un index créé par une version antérieure du kit est migré automatiquement à la première ouverture ; ses documents sont ré-ingérés une fois.
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
