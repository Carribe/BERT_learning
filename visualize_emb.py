from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px


def visualize_clusters(embeddings, texts, clusters, frequencies):
    # Применение T-SNE для снижения размерности эмбеддингов до 2D
    tsne = TSNE(n_components=2, random_state=42, learning_rate=10, perplexity=5)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Создание DataFrame с данными для визуализации
    df_vis = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
    df_vis['Word'] = texts
    df_vis['Cluster'] = clusters
    df_vis['Frequency'] = frequencies  # Добавляем столбец с частотностью

    # Создание интерактивной визуализации с помощью plotly
    fig = px.scatter(df_vis, x='x', y='y', color='Cluster', hover_data=['Word', 'Frequency'],
                     title='Кластеризация слов из файла CSV с помощью BERT и T-SNE',
                     labels={'Cluster': 'Кластер'})

    # Отображение визуализации
    fig.show()