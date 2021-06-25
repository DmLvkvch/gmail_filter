import numpy as np
import yaml
from keras.preprocessing.text import Tokenizer


class ProcessText:

    @staticmethod
    def get_prepared_tokenizer(email_texts: list) -> Tokenizer:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(email_texts)
        return tokenizer

    @staticmethod
    def map_embeddings_to_word_index(embeddings: dict, word_index: dict):
        mapped_embeddings = np.zeros((len(word_index) + 1, 100))
        for word, index in word_index.items():
            emb_word_vector = embeddings.get(word)
            if emb_word_vector is not None:
                mapped_embeddings[index] = emb_word_vector
        return mapped_embeddings

    @staticmethod
    def encode_labels(labels: list) -> list:
        encoded_labels = []
        for label in labels:
            if label == "ham":
                encoded_labels.append(1.0)
            else:
                encoded_labels.append(0.0)

        return encoded_labels

    @staticmethod
    def load_model_config(path: str):
        with open(path) as config_file:
            return yaml.load(config_file, Loader=yaml.FullLoader)
