import prepare
import numpy as np
from keras.preprocessing.sequence import pad_sequences

import data_loader as dl
import model
import prepare
import train

import matplotlib.pyplot as plt


class SpamFilter:

    def __init__(self):
        self.config = spamfilter.prepare.load_model_config('../config.yml')
        self.dataset_path = '../data/collections/SMSSpamCollection.txt'
        self.word_emb_path = '../data/word-embeddings/glove.6B.100d.txt'
        self.model = 0
        self.tokenizer = 0

    def load_weights_from_file(self, path: str, seq_model):
        seq_model.load_weights(path)

    def decode_index(self, index: int) -> str:
        return {0: "ham", 1: "spam"}[index]

    def probability_to_index(self, prediction) -> int:
        return 0 if prediction < 0.5 else 1

    def get_prediction(self, seq_model, tokenizer, text):
        seq = tokenizer.texts_to_sequences([text])
        seq = pad_sequences(seq, maxlen=100)
        prediction = seq_model.predict(seq)
        return prediction

    def predict_text(self, text):
        return self.get_prediction(self.model, self.tokenizer, text)

    def spam_filter(self):
        labels, texts = dl.load_data_from_file(self.dataset_path)
        word_embeddings = dl.load_word_embeddings_from_file(self.word_emb_path)
        tokenizer = prepare.get_prepared_tokenizer(texts)
        self.tokenizer = tokenizer
        embeddings_matrix = prepare.map_embeddings_to_word_index(
            word_embeddings, tokenizer.word_index)
        seq_model = model.get_compiled_model(embeddings_matrix, self.config)
        self.load_weights_from_file('D:\ВКР\spam-filter\checkpoints\weights-improvement-12-0.99.hdf5', seq_model)
        self.model = seq_model


if __name__ == "__main__":
    spam_filter = SpamFilter()
    spam_filter.spam_filter()
