import numpy as np
import tqdm
from fasttext import load_model
import csv
import pandas as pd
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


class DataLoader:

    def __init__(self):
        self.stop_words = stopwords.words('english')
        self.stemmer = SnowballStemmer('english')
        self.text_cleaning_re = "@\S+|https?:\S+|http?:\S+|[^A-Za-z0-9]:\S+|subject:\S+|nbsp"

    def preprocess(self, text, stem=False):
        text = nltk.re.sub(text.text_cleaning_re, ' ', str(text).lower()).strip()
        text = text.replace("we ' re", "we are")
        text = text.replace("they ' re", "they are")
        text = text.replace("you ' re", "you are")
        text = text.replace("Subject:", " ")
        tokens = []
        for token in text.split():
            if token not in self.stop_words:
                if stem:
                    tokens.append(self.stemmer.stem(token))
                else:
                    tokens.append(token)
        return " ".join(tokens)

    @staticmethod
    def load_data_from_file(path: str) -> tuple:
        file = open(path, 'r', encoding="utf8")
        Lines = file.readlines()
        labels = []
        emails = []
        for line in Lines:
            label, email = line.strip().split(maxsplit=1)
            labels.append(label.strip())
            emails.append(email.strip())
        # dataset = pd.read_csv(path)
        # labels = dataset['label'].tolist()
        # dataset.text = dataset.text.apply(lambda x: preprocess(x))
        # emails = dataset['text'].tolist()
        return labels, emails

    @staticmethod
    def load_word_embeddings_from_file(path: str) -> dict:
        # word_embeddings = {}
        # f = load_model(path)
        # words = f.get_words()
        # print(str(len(words)) + " " + str(f.get_dimension()))
        # for w in words:
        #     v = f.get_word_vector(w)
        #     vector = np.asarray(v, dtype='float32')
        #     word_embeddings[w] = vector
        word_embeddings = {}
        with open(path, 'r', encoding='utf8') as word_embeddings_file:
            for line in tqdm.tqdm(word_embeddings_file, "Reading word embeddings"):
                word, *vector = line.split()
                vector = np.asarray(vector, dtype='float32')
                word_embeddings[word] = vector
        return word_embeddings
