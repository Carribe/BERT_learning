Кластеризация слов из CSV-файла с использованием BERT и T-SNE

Данный репозиторий содержит скрипт, демонстрирующий,
как кластеризовать слова из CSV-файла с помощью языковой модели BERT и алгоритма T-SNE.
В скрипте используется библиотека Hugging Face Transformers для загрузки предобученной модели BERT
и получения эмбеддингов слов. Затем применяется T-SNE для снижения размерности эмбеддингов,
и кластеризация проводится с помощью DBSCAN. Результаты кластеризации визуализируются с помощью библиотеки Plotly.

Скрипт генерирует интерактивную визуализацию кластеров слов с помощью библиотеки Plotly.
Каждый кластер будет представлен разным цветом на двумерном графике рассеяния.
При наведении на точку, отобразится всплывающая подсказка с словом и его частотностью.

Скрипт оценивает качество кластеризации с помощью двух метрик: Silhouette Score и Davies-Bouldin Index.
Значения этих метрик будут выведены в консоль.

Не стесняйтесь экспериментировать с различными параметрами и данными.

English

Clustering Words from CSV File using BERT and T-SNE
This repository contains a Python script that demonstrates
how to cluster words from a CSV file using the BERT language model and the T-SNE algorithm.
The script uses the Hugging Face Transformers library to load the pre-trained BERT model
and obtain word embeddings. It then applies T-SNE to reduce the dimensionality of the embeddings
and performs clustering using DBSCAN. The resulting clusters are visualized using Plotly.

The script will generate an interactive visualization of the word clusters using Plotly.
Each cluster will be represented by a different color on the 2D scatter plot.
When hovering over a point, you will see a tooltip displaying the word and its frequency.

Clustering Quality Evaluation
The script evaluates the quality of the clustering using two metrics: Silhouette Score and Davies-Bouldin Index.
The values of these metrics will be printed to the console.

Feel free to experiment with different parameters and data.