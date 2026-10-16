import json
from collections import defaultdict
from pathlib import Path
 
import numpy as np
import requests
from sklearn.cluster import KMeans

 
API_URL = (
    "https://apilux3.pcloud.com/getpublinkdownload"
    "?fileid=93128819642&code=XZnqkRJZ98VUd3rEuMXAFIsEpAtnNHoKYK8k"
    "&linkpassword=&forcedownload=1"
)
FILE_NAME = "objet_consultations_1000.txt"
NUM_CLUSTERS = 5
 
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3-embedding:0.6b"
OLLAMA_BATCH_SIZE = 64
 
ES_URL = "http://localhost:9200"
CLUSTERS_INDEX = "data_clusters"
ES_BULK_SIZE = 500
 
 
def get_data_from_cloud() -> list[str]:
    """Télécharge le fichier depuis pCloud (s'il est absent) et retourne ses lignes non vides."""
    file_path = Path(FILE_NAME)
 
    if not file_path.exists():
        # L'API pCloud renvoie un chemin et une liste d'hôtes pour construire le lien final
        response = requests.get(API_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
        download_link = f"https://{data['hosts'][0]}{data['path']}"
        print("Download Link:", download_link)
 
        file_response = requests.get(download_link, timeout=60)
        file_response.raise_for_status()
        file_path.write_bytes(file_response.content)
 
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
 
    print(f"Loaded {len(lines)} lines from {FILE_NAME}.")
    return lines
 
 
def get_data_embeddings(lines: list[str]) -> np.ndarray:
    
    """Calcule un embedding pour chaque ligne en interrogeant le docker Ollama (par lots)."""
    print(f"Calculating embeddings with Ollama ({OLLAMA_MODEL})...")
    embeddings: list[list[float]] = []
 
    for start in range(0, len(lines), OLLAMA_BATCH_SIZE):
        batch = lines[start : start + OLLAMA_BATCH_SIZE]
        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": "qwen3-embedding:0.6b", "input": batch},
            timeout=120,
        )
        response.raise_for_status()
        embeddings.extend(response.json()["embeddings"])
        print(f"  {min(start + OLLAMA_BATCH_SIZE, len(lines))}/{len(lines)}", end="\r")
 
    result = np.array(embeddings, dtype=np.float32)
    print(f"\nSuccessfully generated embeddings! Shape: {result.shape}")
    return result
 
 
def cluster_data(
    embeddings: np.ndarray, lines: list[str], num_clusters: int = NUM_CLUSTERS
) -> tuple[dict[int, list[str]], dict[int, np.ndarray], np.ndarray]:
    """Regroupe les lignes par K-means.
 
    Retourne :
      - {cluster: [lignes]}
      - {cluster: matrice des embeddings des lignes du cluster}
      - les centroïdes (un par cluster), utilisés comme embedding du cluster
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(embeddings)
 
    cluster_to_lines: dict[int, list[str]] = defaultdict(list)
    cluster_to_embeddings: dict[int, list[np.ndarray]] = defaultdict(list)
    for line, embedding, label in zip(lines, embeddings, cluster_labels):
        cluster_to_lines[int(label)].append(line)
        cluster_to_embeddings[int(label)].append(embedding)
 
    return (
        dict(sorted(cluster_to_lines.items())),
        {k: np.array(v) for k, v in sorted(cluster_to_embeddings.items())},
        kmeans.cluster_centers_,
    )
 
 
# ---------------------------------------------------------------------------
# Elasticsearch
# ---------------------------------------------------------------------------
def recreate_index(index: str, dims: int, extra_properties: dict) -> None:
    """Supprime puis recrée un index avec un champ embedding de type dense_vector."""
    requests.delete(f"{ES_URL}/{index}", timeout=30)  # 404 ignoré si l'index n'existe pas
 
    mapping = {
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
                **extra_properties,
            }
        }
    }
    response = requests.put(f"{ES_URL}/{index}", json=mapping, timeout=30)
    response.raise_for_status()
 
 
def bulk_index(index: str, documents: list[dict]) -> None:
    """Indexe une liste de documents via l'API _bulk, par paquets."""
    for start in range(0, len(documents), ES_BULK_SIZE):
        chunk = documents[start : start + ES_BULK_SIZE]
        payload = "".join(
            json.dumps({"index": {"_index": index}}) + "\n" + json.dumps(doc) + "\n"
            for doc in chunk
        )
        response = requests.post(
            f"{ES_URL}/_bulk",
            data=payload.encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
            timeout=120,
        )
        response.raise_for_status()
        if response.json().get("errors"):
            raise RuntimeError(f"Erreur lors de l'indexation dans '{index}': {response.text[:500]}")
 
    requests.post(f"{ES_URL}/{index}/_refresh", timeout=30).raise_for_status()
 
 
def index_clusters(
    cluster_to_lines: dict[int, list[str]],
    cluster_to_embeddings: dict[int, np.ndarray],
    centroids: np.ndarray,
) -> None:
    """Crée un index par cluster (data_1, data_2...) + l'index data_clusters."""
    dims = centroids.shape[1]
 
    recreate_index(CLUSTERS_INDEX, dims, {"data_index": {"type": "keyword"}})
    cluster_documents = []
 
    for cluster, cluster_lines in cluster_to_lines.items():
        index_name = f"data_{cluster + 1}"
 
        recreate_index(index_name, dims, {"line": {"type": "text"}})
        bulk_index(
            index_name,
            [
                {"embedding": embedding.tolist(), "line": line}
                for line, embedding in zip(cluster_lines, cluster_to_embeddings[cluster])
            ],
        )
        print(f"Indexed {len(cluster_lines)} documents in '{index_name}'.")
 
        cluster_documents.append(
            {"embedding": centroids[cluster].tolist(), "data_index": index_name}
        )
 
    bulk_index(CLUSTERS_INDEX, cluster_documents)
    print(f"Indexed {len(cluster_documents)} clusters in '{CLUSTERS_INDEX}'.")
 
 
def main() -> None:
    lines = get_data_from_cloud()
    line_embeddings = get_data_embeddings(lines)
    cluster_to_lines, cluster_to_embeddings, centroids = cluster_data(line_embeddings, lines)
 
    print("\nNumber of lines per cluster:")
    for cluster, cluster_lines in cluster_to_lines.items():
        print(f"  Cluster {cluster}: {len(cluster_lines)}")
 
    print("\n--- Examples from each cluster ---")
    for cluster, cluster_lines in cluster_to_lines.items():
        print(f"\nCluster {cluster}:")
        for line in cluster_lines[:3]:
            print(f" - {line}")
 
    print("\n--- Indexing into Elasticsearch ---")
    index_clusters(cluster_to_lines, cluster_to_embeddings, centroids)
 
 
if __name__ == "__main__":
    main()