"""Reranking par LLM local (Ollama).

Pour chaque document, on demande au modèle si le document répond à la requête (yes / no).
On lit les logprobs du token de la réponse, on additionne les probabilités des variantes de
"yes" et de "no" (yes / Yes / YES / " yes"...), puis on calcule :

    score = p_yes / (p_yes + p_no)

Les documents sont enfin triés par score décroissant.

Remarque : avec une sortie contrainte par un schéma JSON, certaines versions d'Ollama ne
renvoient les logprobs que pour le tout premier token généré. On essaie donc plusieurs
stratégies (voir STRATEGIES) et on garde la première qui fournit les logprobs yes/no.
"""

import json
import logging
import math
import os
import re

import requests
from concurrent.futures import ThreadPoolExecutor

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3.5:0.8b")

# Nombre de demandes envoyées en parallèle à Ollama (une demande par document)
RERANK_WORKERS = int(os.getenv("RERANK_WORKERS", "4"))

# Nombre d'alternatives demandées à Ollama pour chaque token
TOP_LOGPROBS = 20

logger = logging.getLogger("rerank")

# Stratégie 1 ("object") : sortie contrainte au JSON {"answer": "yes" | "no"}
OBJECT_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["answer"],
}
# Stratégie 2 ("string") : sortie contrainte à l'enum JSON "yes" | "no" (le yes/no arrive
# dès le premier token, là où Ollama fournit toujours les logprobs)
STRING_SCHEMA = {"type": "string", "enum": ["yes", "no"]}

SYSTEM_PROMPTS = {
    "object": (
        "Judge whether the Document is relevant to the search Query. "
        'Answer only with JSON of the form {"answer": "yes"} or {"answer": "no"}.'
    ),
    "string": (
        "Judge whether the Document is relevant to the search Query. "
        'Answer only with "yes" or "no".'
    ),
    "free": (
        "Judge whether the Document is relevant to the search Query. "
        "Answer with a single word: yes or no."
    ),
}

# Ordre d'essai : la contrainte JSON demandée d'abord, puis les variantes de repli
STRATEGIES = ["object", "string", "free"]

# Stratégie retenue après le premier document (None = pas encore déterminée)
_active_strategy: str | None = None


def _normalize(token: str) -> str:
    """' Yes', 'YES', '"yes' -> 'yes' : on ignore espaces, guillemets et casse."""
    return token.strip().strip('"').strip().lower()


def _p_yes_no(alternatives: list[dict]) -> tuple[float, float]:
    """Additionne les probabilités des variantes de 'yes' et de 'no'."""
    p_yes = p_no = 0.0
    for alt in alternatives:
        word = _normalize(alt["token"])
        probability = math.exp(alt["logprob"])
        if word == "yes":
            p_yes += probability
        elif word == "no":
            p_no += probability
    return p_yes, p_no


_ANSWER_PREFIX = re.compile(r'"answer"\s*:\s*"')


def _answer_token_alternatives(content: str, logprobs: list[dict]) -> list[dict] | None:
    """Stratégie "object" : retourne les alternatives du token qui porte la valeur de "answer".

    On reconstruit le texte token par token pour repérer le token qui contient le premier
    caractère de la valeur (juste après `"answer": "`). Si Ollama ne fournit pas les logprobs
    de ce token (cas observé : seul le 1er token `{"` en a), on retourne None.
    """
    match = _ANSWER_PREFIX.search(content)
    if not match:
        return None
    value_start = match.end()

    offset = 0
    for entry in logprobs:
        end = offset + len(entry["token"])
        if offset <= value_start < end:
            return entry.get("top_logprobs") or [entry]
        offset = end
    return None


def yes_no_from_logprobs(logprobs: list[dict], content: str = "", strategy: str = "string") -> tuple[float, float] | None:
    """Retourne (p_yes, p_no) lues sur le token de la réponse, ou None s'il n'est pas disponible.

    - "object" : le token de la réponse est au milieu du JSON (voir _answer_token_alternatives) ;
      les alternatives du 1er token `{"` ne comptent pas (yes/no y sont des probabilités parasites).
    - "string" / "free" : le token de la réponse est le tout premier token généré.
    """
    if strategy == "object":
        alternatives = _answer_token_alternatives(content, logprobs)
    else:
        alternatives = (logprobs[0].get("top_logprobs") or [logprobs[0]]) if logprobs else None
    if not alternatives:
        return None

    p_yes, p_no = _p_yes_no(alternatives)
    return (p_yes, p_no) if p_yes + p_no > 0 else None


def ask_ollama(query: str, document: str, strategy: str = "object", model: str | None = None) -> dict:
    """Appelle /api/chat avec logprobs. Retourne {"content": ..., "logprobs": [...]}.

    Le prompt commence par la partie commune à tous les documents (instructions + question) et
    finit par le document : le début identique est réutilisé par le cache clé-valeur (KV cache)
    d'Ollama d'une demande à l'autre.
    """
    payload = {
        "model": model or RERANK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[strategy]},
            {"role": "user", "content": f"Query: {query}\nDocument: {document}"},
        ],
        "stream": False,
        "think": False,  # pas de phase de réflexion : on veut directement la réponse
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        "options": {"temperature": 0},
    }
    if strategy == "object":
        payload["format"] = OBJECT_SCHEMA  # sortie contrainte au JSON yes/no
    elif strategy == "string":
        payload["format"] = STRING_SCHEMA  # sortie contrainte à l'enum "yes" | "no"
    else:
        payload["options"]["num_predict"] = 3  # pas de contrainte : on ne lit que le 1er token

    response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    body = response.json()
    return {
        "content": body.get("message", {}).get("content", ""),
        "logprobs": body.get("logprobs") or [],
    }


def _binary_answer(content: str) -> float:
    """Repli sans logprobs : 1.0 si la réponse est 'yes', sinon 0.0."""
    try:
        parsed = json.loads(content)
        answer = parsed.get("answer") if isinstance(parsed, dict) else parsed
    except json.JSONDecodeError:
        answer = content
    return 1.0 if isinstance(answer, str) and _normalize(answer).startswith("yes") else 0.0


def score_document(query: str, document: str, model: str | None = None) -> float:
    """Score de pertinence p_yes / (p_yes + p_no) d'un document pour une requête (0 à 1)."""
    global _active_strategy

    # Une fois la stratégie trouvée, on l'utilise directement ; sinon on les essaie dans l'ordre
    strategies = [_active_strategy] if _active_strategy else STRATEGIES
    last_content = ""

    for strategy in strategies:
        body = ask_ollama(query, document, strategy, model)
        last_content = body["content"] if strategy == strategies[0] else last_content
        probabilities = yes_no_from_logprobs(body["logprobs"], body["content"], strategy)
        if probabilities:
            p_yes, p_no = probabilities
            if _active_strategy is None:
                _active_strategy = strategy
                logger.info("Stratégie de reranking retenue : %s", strategy)
            return p_yes / (p_yes + p_no)

    # Aucune stratégie ne renvoie de logprobs yes/no : réponse binaire
    if _active_strategy is None:
        _active_strategy = STRATEGIES[0]
        logger.warning("Aucun logprob yes/no renvoyé par Ollama : scores binaires (0 ou 1)")
    return _binary_answer(last_content)


def rerank(query: str, documents: list[str], top_n: int | None = None, model: str | None = None) -> list[dict]:
    """Reclasse 'documents' selon leur pertinence pour 'query', score décroissant.

    Chaque document est soumis au modèle dans une demande séparée ; les demandes partent en
    parallèle (RERANK_WORKERS à la fois). Les lignes en double ne sont notées qu'une fois.

    Retourne [{"id": position du document dans la liste d'entrée, "line": texte,
    "rerank_score": p_yes / (p_yes + p_no)}, ...] (limité à top_n si fourni).
    """
    unique = list(dict.fromkeys(documents))
    scores: dict[str, float] = {}

    # Tant que la stratégie n'est pas déterminée, le premier document est traité seul
    # (sinon toutes les demandes parallèles referaient la découverte en même temps).
    if unique and _active_strategy is None:
        scores[unique[0]] = score_document(query, unique[0], model)
        unique = unique[1:]

    if unique:
        with ThreadPoolExecutor(max_workers=max(1, min(RERANK_WORKERS, len(unique)))) as pool:
            for doc, score in zip(unique, pool.map(lambda d: score_document(query, d, model), unique)):
                scores[doc] = score

    scored = [
        {"id": index, "line": doc, "rerank_score": scores[doc]}
        for index, doc in enumerate(documents)
    ]
    scored.sort(key=lambda item: item["rerank_score"], reverse=True)
    return scored[:top_n] if top_n else scored


if __name__ == "__main__":
    # Petit test manuel : python rerank.py
    docs = [
        "Lave linge séchant",
        "MARCHE DE SERVICE D'ASSURANCES",
        "Assurance \"Responsabilité et risques annexes\"",
        "RÉFECTION DU MUR SUD DU CIMETIÈRE",
    ]
    for item in rerank("assurance", docs):
        print(f"{item['rerank_score']:.3f}  (id {item['id']})  {item['line']}")
    print("Stratégie utilisée :", _active_strategy)
