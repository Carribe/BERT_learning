from transformers import BertTokenizer, BertModel
import torch


def process_text_data(texts):
    # Загрузка предварительно обученного токенайзера BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Загрузка предварительно обученной модели BERT
    model = BertModel.from_pretrained('bert-base-multilingual-cased')

    # Токенизация текстов и преобразование в тензоры с паддингом
    encoded_texts = tokenizer(texts, padding='max_length', truncation=True, return_tensors='pt', max_length=32)

    # Передача данных через модель BERT для получения эмбеддингов
    with torch.no_grad():
        model_outputs = model(**encoded_texts)

    # Извлечение эмбеддингов из выходных данных модели BERT
    embeddings = model_outputs.last_hidden_state  # Последний скрытый слой содержит эмбеддинги

    # Преобразование списка тензоров эмбеддингов в один большой тензор фиксированной длины
    embeddings_tensor = torch.cat([emb.mean(dim=0, keepdim=True) for emb in embeddings], dim=0)

    return embeddings_tensor
