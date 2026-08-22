"""Query rewrite + optional HyDE via a small local Ollama model."""

from __future__ import annotations

import json
import re

from .embed import OllamaError, generate

REWRITE_PROMPT = """Tu reformules une question pour la recherche documentaire.
Réponds UNIQUEMENT avec un JSON valide, sans markdown :
{{"queries": ["...", "...", "..."], "hyde": "court paragraphe hypothétique"}}
- 2 ou 3 reformulations courtes, dans la langue de la question
- hyde = réponse plausible de 2-4 phrases (pas une vraie source)

Question: {question}
"""

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string", "minLength": 3},
            "minItems": 2,
            "maxItems": 3,
        },
        "hyde": {"type": "string", "minLength": 20},
    },
    "required": ["queries", "hyde"],
    "additionalProperties": False,
}


def rewrite_query(host: str, model: str, question: str, hyde: bool = True) -> dict:
    original = question.strip()
    fallback = {"queries": [original], "hyde": ""}
    try:
        raw = generate(
            host,
            model,
            REWRITE_PROMPT.format(question=question),
            timeout=60.0,
            json_schema=REWRITE_SCHEMA,
            think=False,
            num_predict=320,
        )
    except OllamaError:
        return fallback
    blob = raw
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        blob = m.group(0)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return fallback
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        return fallback

    queries = [original]
    for candidate in data["queries"]:
        if not isinstance(candidate, str):
            continue
        q = candidate.strip()
        if q and q not in queries:
            queries.append(q)
        if len(queries) == 4:
            break
    # A valid expansion is the original plus at least two unique rephrases.
    # Anything shorter is treated as a schema-quality failure and degrades cleanly.
    if len(queries) < 3:
        return fallback

    hyde_text = ""
    if hyde:
        candidate_hyde = data.get("hyde")
        if not isinstance(candidate_hyde, str) or not candidate_hyde.strip():
            return fallback
        hyde_text = candidate_hyde.strip()
    return {"queries": queries, "hyde": hyde_text}
