# RAG portable — Brainstorm

Document de design. Distinct du plan d’implémentation (`.hermes/plans/…-rag-portable.md`). Les décisions de la section 2 sont closes.

## 1. Contexte et but

Kit RAG **local, généralisable, portable**. On copie le dossier `RAG/` dans n’importe quel dossier de travail, à côté des documents. Un humain ou un agent futur n’appelle que deux scripts (`ingest.py` / `search.py`, JSON sur stdout). Pas d’API, pas d’agents dans le kit.

```text
<workdir>/
  docs/              ← corpus (.md .txt .rst .org .pdf)
  RAG/               ← kit copié (config, scripts, indexes SQLite)
    ingest.py
    search.py
    config.yaml
    indexes/         ← un .sqlite par collection
```

**Workdir** = parent du dossier `RAG/` (là où vit `docs/`).
**Collections** = noms de dossiers, pas des domaines métier.

Repo unique : `~/Build/RAG_build/`. `~/Build/exemple/` a été **supprimé** ; ne pas le recréer. Le code vit uniquement ici. L’ébauche Python existe déjà (`RAG/rag/*.py`) ; ce doc fige le *quoi*, pas le *comment* des tâches.

Cible : retrieval utile sur un corpus de travail (notes, PDF, markdown), 100 % offline une fois Ollama et le venv en place.

## 2. Décisions figées (ne pas rouvrir)

1. **100 % local.** Ollama à `http://127.0.0.1:11434`. **Pas Docker**, pas Qdrant, pas de service distant.
2. **Embedding dense unique** : `embeddinggemma` (même modèle à l’index **et** à la query). Pas d’embedder « query-only ».
3. **Retrieval hybride** = dense (Gemma, cosine L2-norm) **+ BM25 lexical** (SQLite FTS5). **Pas** de sparse appris (SPLADE). **Pas** de late interaction (ColBERT).
4. **Pas de rerank.** Ni cross-encoder MiniLM / sentence-transformers / torch, ni pairwise Ollama (`rerank: ollama`). Fusion **RRF uniquement** (éventuellement un léger bonus dense pour ne pas noyer un hit vectoriel isolé). L’ébauche qui expose `--rerank ollama` / `_ollama_rerank` doit s’aligner : à retirer, pas à étendre.
5. **Query expansion** : la question **originale est toujours gardée**, plus 2–3 reformulations, plus un paragraphe HyDE, le tout via `qwen3.5:0.8b`. `--no-rewrite` et fallback si Ollama down / JSON cassé → `queries == [question]`, la search continue.
6. **Chunk défaut `parent_child`** : child ~256 tokens **indexé** (dense + FTS), parent ~1024 **renvoyé** dans `hits[].text`. Overlap ~32. Alternatives : `section` (bloc entier, locator = titre / `Article …`), `window` (fenêtre glissante, pas de locator). Pas de semantic chunking, late-chunking, RAPTOR.
7. **Métadonnées par hit** : `path`, `page` (PDF, sinon `null`), `locator` (titre markdown / `Article …`), `collection`. Pas de champs métier (juridiction, compte, dossier, etc.).
8. **Pas d’agents dans le kit.** Les agents futurs (juridique, comptable, …) appelleront uniquement les scripts. Le kit ne contient ni prompts d’agent, ni orchestrateur, ni types métier.
9. **Pas de types métier** dans config, chunk, store ou JSON. Un dossier de docs = une collection nommée.
10. **Code ici seulement.** Pas de second arbre `exemple/`. Fixture de test éventuelle **dans** le repo (`RAG/tests/fixtures/`), pas un projet externe.

## 3. Pipeline ingest et query

```text
INGEST
  <workdir>/docs  (ou --path)
       │
       ├─ textio : .md/.txt/.rst/.org → 1 page (page=null)
       │           .pdf               → 1 page numérotée / feuille
       ├─ chunk  : parent_child | section | window
       │           child.text indexé ; parent_text conservé
       ├─ embed  : POST /api/embed  embeddinggemma  (batch)
       └─ store  : SQLite
                   documents(path, sha, collection)
                   chunks(text, parent_text, locator, page, kind, embedding)
                   chunks_fts (FTS5, BM25) sur le texte child

QUERY
  question
       │
       ├─ rewrite (sauf --no-rewrite) : qwen3.5:0.8b
       │     queries = [originale, rephrase…]   (max 4)
       │     hyde    = paragraphe hypothétique  (optionnel)
       │     fallback : queries = [question], hyde = ""
       ├─ dense : embed Gemma des queries (+ hyde) → cosine vs children
       ├─ BM25  : FTS5 MATCH par query, tokens FR (accents / apostrophes)
       ├─ fusion RRF (listes dense + FTS) → top-k
       └─ hit   : text = parent (sinon child)
                  + score, path, page, locator, collection
                  JSON unique sur stdout
```

Indexation et requête **partagent** le même embedder. Le child sert à matcher ; le parent sert à lire.

## 4. Outils

| Outil | Rôle | Notes |
|---|---|---|
| Ollama `127.0.0.1:11434` | embeddings + rewrite/HyDE | `embeddinggemma:latest`, `qwen3.5:0.8b` |
| Python 3.11 + `RAG/.venv` | runtime isolé | bootstrap auto si le venv existe |
| `pyyaml` `numpy` `pypdf` | config, vecteurs, PDF | `requirements.txt` |
| SQLite + FTS5 | store portable (fichiers) | un `.sqlite` / collection sous `RAG/indexes/` |
| `urllib` (stdlib) | client HTTP Ollama | pas de SDK cloud |
| pytest | tests chunk / store / scripts | à ajouter ; pas dans le kit runtime |
| Git local LeVibeux | historique | pas de remote sauf demande |
| **Pas Docker** | — | décidé non |
| **Pas torch / MiniLM / ColBERT** | — | hors scope |

## 5. Contrat scripts (JSON)

Appel depuis le **workdir** (parent de `RAG/`). Stdout = **un** objet JSON. Stderr pour traces humaines si besoin ; un agent ne parse que stdout.

### `ingest.py`

```bash
python RAG/ingest.py
python RAG/ingest.py --path ../docs
python RAG/ingest.py --path … --collection default
python RAG/ingest.py --config RAG/config.yaml
```

Succès (exit 0) :

```json
{
  "ok": true,
  "collection": "default",
  "docs_dir": "/abs/workdir/docs",
  "files": 3,
  "chunks": 42,
  "index": "/abs/workdir/RAG/indexes/default.sqlite",
  "stats": { "documents": 3, "chunks": 42 }
}
```

Échec corpus vide / erreur (exit 2) : `{ "ok": false, "error": "…" }`.

Comportement : un fichier déjà connu (même `path`) est **remplacé** (sha + chunks). Formats lus : markdown, texte, rst, org, PDF (page 1-N).

### `search.py`

```bash
python RAG/search.py "question"
python RAG/search.py --query "question" --k 8
python RAG/search.py --query "question" --no-rewrite
python RAG/search.py --query "question" --collection default
```

Pas de flag `--rerank`. Pas de mode `ollama`.

Succès avec hits (exit 0) :

```json
{
  "ok": true,
  "query": "…",
  "rewritten": ["…", "…"],
  "hyde": "…",
  "collection": "default",
  "hits": [
    {
      "text": "passage parent",
      "score": 0.81,
      "path": "../docs/note.md",
      "page": null,
      "locator": "Titre de section",
      "collection": "default"
    }
  ]
}
```

- `page` : entier PDF ou `null`.
- `locator` : titre / `Article …` ou `null` (`window`).
- `text` : parent (~1024) en `parent_child` ; sinon le chunk lui-même.
- Index vide ou query manquante : `{ "ok": false, "error": "…" }` (exit 2).
- Index OK, zéro hit : `ok: true`, `hits: []` (exit 1).

Le kit ne parle pas aux agents : il imprime ce JSON. Point.

## 6. Hors scope

- Docker, Qdrant, Docling, Streamlit, FastAPI, LangChain
- Agents (juridique, comptable, …) et prompts métier
- MiniLM, sentence-transformers, torch, `rerank: ollama`
- Sparse appris (SPLADE) et late interaction (ColBERT)
- Late chunking, RAPTOR, GraphRAG, contextual retrieval
- Types métier dans les métadonnées
- Dossier `exemple/` (supprimé, ne pas recréer)
- GitHub remote / push
- Modèles 7B+ / 14B / 27B dans la boucle RAG

## 7. Risques machine (i5-1135G7, ~15 Go RAM, pas de GPU)

| Risque | Effet | Mitigation design |
|---|---|---|
| Ollama déjà occupé par un gros GGUF | timeout embed / rewrite | erreur claire `OllamaError` ; `--no-rewrite` ; ne pas enchaîner 7B+ et le RAG |
| Rewrite 0.8b + embed Gemma en parallèle d’un autre modèle | RAM saturée, swap | un seul petit modèle de rewrite ; jamais 14B/27B dans ce kit |
| FTS5 `MATCH` + ponctuation FR (apostrophes, accents) | crash ou 0 hit lexical | tokenizer alphanum + accents **avant** MATCH ; ne pas passer la question brute |
| PDF scannés / sans couche texte | pages vides → 0 chunk | pas d’OCR (hors scope) ; ingest saute le fichier vide |
| Index entier en RAM (`all_vectors`) | OK pour petits corpus, limite plus tard | V1 assume un workdir de notes, pas un dump légal de 10 Go |
| Copie du kit oubliée / mauvais `docs_path` | ingest d’un autre arbre | résoudre `../docs` et `indexes` depuis le dossier `RAG/` (parent du `config.yaml`) ; workdir = ce parent |

Ne jamais charger un 14B/27B pour reranker ou réécrire. Le 0.8b est le plafond de génération du kit.

## 8. Questions ouvertes (mineures, ne bloquent pas V1)

- **Distribution du kit** : V1 = **copie du dossier `RAG/`**. Submodule / git-subtree plus tard si ça gêne vraiment.
- **Chemin affiché dans `hits[].path`** : relatif au dossier `RAG/` (ébauche actuelle) vs relatif au workdir. À trancher à l’usage agent ; les deux restent du string path, pas un type métier.
- **Pool RRF vs `k`** : ébauche `pool=24`, `k=8`. Suffisant en V1 ; pas besoin d’exposer `pool` au CLI.
- **Rerank** : **non en V1**. Ne pas le réintroduire sans demande explicite.

Fin.
