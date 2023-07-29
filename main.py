import pandas as pd
from sklearn.cluster import DBSCAN
from visualize_emb import visualize_clusters
from text_processor import process_text_data
from evaluate_clustering_quality import evaluate_clustering_quality


def main():
    # Загрузка данных из CSV файла
    csv_file = "/dir/file.csv"
    df = pd.read_csv(csv_file, sep=",", encoding="utf-8", skiprows=0, nrows=300)

    # Получение столбца Word (слова) и Frequency (частотность) из DataFrame
    texts = df['Word'].tolist()
    frequencies = df['Frequency'].tolist()

    # Обработка текстовых данных и получение эмбеддингов
    embeddings = process_text_data(texts)

    # Преобразование тензоров эмбеддингов в массив NumPy
    embeddings_np = embeddings.numpy()

    # Кластеризация эмбеддингов с помощью DBSCAN
    db_scan = DBSCAN(eps=7, min_samples=3)
    clusters = db_scan.fit_predict(embeddings_np)

    # Оценка качества кластеризации с помощью силуэта и индекса Данна
    evaluate_clustering_quality(embeddings_np, clusters)

    # Визуализация кластеров с помощью plotly
    visualize_clusters(embeddings_np, texts, clusters, frequencies)

if __name__ == "__main__":
    main()
