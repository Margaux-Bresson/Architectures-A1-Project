import os
import uuid

import requests
from elasticsearch import ApiError, Elasticsearch, NotFoundError, TransportError
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rerank import RERANK_MODEL, rerank

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-embedding:0.6b")
CLUSTERS_INDEX = "data_clusters"

# Nombre minimum de candidats récupérés par la recherche hybride avant le reranking
# (on reclasse plus de lignes que demandé, puis on garde les k meilleures)
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))

# Pondération de la recherche hybride (voir la doc Elastic "knn query in hybrid search") :
# les scores BM25 (match) ne sont pas bornés alors que le score kNN cosinus est dans [0, 1],
# d'où un boost plus faible sur le match pour ne pas écraser la partie vectorielle.
HYBRID_KNN_BOOST = float(os.getenv("HYBRID_KNN_BOOST", "1.0"))
HYBRID_MATCH_BOOST = float(os.getenv("HYBRID_MATCH_BOOST", "0.1"))

es = Elasticsearch(ES_URL)
app = FastAPI()

# CORS : autorise un front servi depuis localhost / 127.0.0.1, sur n'importe quel port
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Texte de la recherche")
    k: int = Field(10, ge=1, le=100, description="Nombre de résultats")
    rerank: bool = Field(True, description="Reclasser les résultats avec le modèle de reranking")


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Texte de la recherche")
    lines: list[str] = Field(..., min_length=1, description="Lignes à reclasser")


class V1RerankRequest(BaseModel):
    """Corps de la requête, au format de l'API de rerank de vLLM / Cohere."""

    model: str | None = Field(None, description="Modèle de reranking (par défaut : RERANK_MODEL)")
    query: str = Field(..., min_length=1, description="La question")
    documents: list[str] = Field(..., min_length=1, description="Les documents à reclasser")
    top_n: int | None = Field(None, ge=1, description="Nombre de documents à retourner")


def embed(text: str) -> list[float]:
    """Calcule l'embedding d'un texte avec Ollama (même modèle que pour l'indexation)."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": OLLAMA_MODEL, "input": text},
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Erreur Ollama : {exc}")
    return response.json()["embeddings"][0]


def knn_search(index: str, vector: list[float], k: int, source: list[str]) -> list[dict]:
    """Recherche kNN sur le champ 'embedding' d'un index et retourne les hits."""
    try:
        result = es.search(
            index=index,
            knn={
                "field": "embedding",
                "query_vector": vector,
                "k": k,
                "num_candidates": max(50, k * 10),
            },
            source=source,
            size=k,
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Index '{index}' introuvable")
    except (ApiError, TransportError) as exc:
        raise HTTPException(status_code=502, detail=f"Erreur Elasticsearch ({index}) : {exc}")
    return result["hits"]["hits"]


def hybrid_search(index: str, query: str, vector: list[float], k: int, source: list[str]) -> list[dict]:
    """Recherche hybride dans un index data_X : requête knn + requête match (BM25) combinées
    dans un bool/should. Les scores des deux clauses (pondérés par leur boost) sont additionnés."""
    try:
        result = es.search(
            index=index,
            query={
                "bool": {
                    "should": [
                        # Partie lexicale : mots de la requête présents dans le champ texte 'line'
                        {"match": {"line": {"query": query, "boost": HYBRID_MATCH_BOOST}}},
                        # Partie vectorielle : proximité sémantique sur le champ 'embedding'
                        {
                            "knn": {
                                "field": "embedding",
                                "query_vector": vector,
                                "k": k,
                                "num_candidates": max(50, k * 10),
                                "boost": HYBRID_KNN_BOOST,
                            }
                        },
                    ]
                }
            },
            source=source,
            size=k,
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Index '{index}' introuvable")
    except (ApiError, TransportError) as exc:
        raise HTTPException(status_code=502, detail=f"Erreur Elasticsearch ({index}) : {exc}")
    return result["hits"]["hits"]


def search_in_index(
    index: str,
    query: str,
    k: int,
    vector: list[float] | None = None,
    use_rerank: bool = True,
) -> dict:
    """Cherche les k meilleures lignes pour 'query' dans un index data_X.

    1) recherche hybride (knn + match) -> candidats
    2) reranking des candidats avec Ollama (p_yes / (p_yes + p_no)), puis on garde les k meilleurs
    """
    vector = vector or embed(query)
    candidates = max(k, RERANK_CANDIDATES) if use_rerank else k
    hits = hybrid_search(index, query, vector, candidates, ["line"])
    lines = [h["_source"]["line"] for h in hits]

    response = {"query": query, "cluster": index, "mode": "hybrid"}

    if use_rerank and lines:
        try:
            ranked = rerank(query, lines, top_n=k)
        except requests.RequestException as exc:
            # Ollama injoignable ou modèle absent : on garde l'ordre de la recherche hybride
            response["rerank_error"] = f"Reranking indisponible : {exc}"
        else:
            response["mode"] = "hybrid+rerank"
            response["results"] = [
                {"line": item["line"], "score": item["rerank_score"]} for item in ranked
            ]
            return response

    response["results"] = [
        {"line": h["_source"]["line"], "score": h["_score"]} for h in hits[:k]
    ]
    return response


@app.get("/")
def read_root() -> dict:
    return {"message": "AI Search backend is running"}


@app.post("/rerank")
def rerank_lines(request: RerankRequest) -> list[dict]:
    """Reclasse une liste de lignes avec le modèle de reranking (Ollama).

    Retourne [{"id": position dans 'lines', "line": ..., "rerank_score": ...}] par score décroissant.
    """
    try:
        return rerank(request.query, request.lines)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Erreur Ollama (reranking) : {exc}")


@app.post("/v1/rerank")
def v1_rerank(request: V1RerankRequest) -> dict:
    """Endpoint dédié au rerank, au format de l'API Cohere / vLLM.

    La question et chaque document sont soumis au modèle dans des demandes séparées, en
    parallèle ; on collecte les probabilités yes/no pour calculer le score de pertinence.
    """
    model = request.model or RERANK_MODEL
    try:
        ranked = rerank(request.query, request.documents, request.top_n, model)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Erreur Ollama (reranking) : {exc}")

    return {
        "id": f"rerank-{uuid.uuid4().hex}",
        "model": model,
        "results": [
            {
                "index": item["id"],
                "document": {"text": item["line"]},
                "relevance_score": item["rerank_score"],
            }
            for item in ranked
        ],
    }


@app.post("/search")
def search_data(request: SearchRequest) -> dict:
    """Recherche en 2 étapes : data_clusters -> champ 'data_index' -> index data_X."""
    vector = embed(request.query)

    # 1) Index data_clusters : recherche vectorielle (knn) du cluster le plus proche -> champ "data_index"
    cluster_hits = knn_search(CLUSTERS_INDEX, vector, 1, ["data_index"])
    if not cluster_hits:
        raise HTTPException(status_code=404, detail="Aucun cluster trouvé (index vide ?)")
    data_index = cluster_hits[0]["_source"]["data_index"]

    # 2) Index data_X renvoyé par la première recherche : recherche hybride (knn + match) puis reranking
    return search_in_index(data_index, request.query, request.k, vector, request.rerank)


@app.get("/search/{cluster_id}")
def search_by_cluster(
    cluster_id: int = Path(..., ge=1, description="Numéro du cluster (1 pour data_1, 2 pour data_2...)"),
    q: str = Query(..., min_length=1, description="Texte de la recherche"),
    k: int = Query(10, ge=1, le=100, description="Nombre de résultats"),
) -> dict:
    """Recherche directement dans l'index d'un cluster donné (sans passer par data_clusters)."""
    return search_in_index(f"data_{cluster_id}", q, k)