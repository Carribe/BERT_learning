from sklearn.metrics import silhouette_score, davies_bouldin_score


def evaluate_clustering_quality(embeddings, clusters):
    silhouette = silhouette_score(embeddings, clusters)
    davies_bouldin = davies_bouldin_score(embeddings, clusters)
    print(f"Оценка силуэта: {silhouette}")
    print(f"Индекс Давида-Боулдина: {davies_bouldin}")
