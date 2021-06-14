import prepare

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, GlobalMaxPool1D, \
    Bidirectional
from tensorflow.keras.optimizers import RMSprop, Adam
import tensorflow as tf


class ModelBuilder:

    def __init__(self):
        print('model')

    @staticmethod
    def get_compiled_model(embeddings_matrix: dict, config: dict) -> Sequential:
        model = tf.keras.Sequential()
        embedding_layer = Embedding(len(embeddings_matrix),
                                    config['embedding']['num_nodes'],
                                    trainable=False,
                                    input_length=config['embedding']['input_len'],
                                    weights=[embeddings_matrix])
        rmsprop = RMSprop(learning_rate=0.001, rho=0.9)
        model.add(embedding_layer)
        # model.add(GlobalMaxPool1D())
        # model.add(Dense(10, activation='relu'))
        # model.add(Conv1D(32, 8, activation="relu"))
        # model.add(MaxPooling1D(2))
        # model.add(Flatten())
        # model.add(Dense(50, activation = "relu"))
        # model.add(Dropout(0.3, noise_shape=None, seed=None))
        # model.add(Dense(50, activation = "relu"))
        # model.add(Dropout(0.2, noise_shape=None, seed=None))
        # model.add(Dense(10, activation = "relu"))
        # model.add(Dense(2, activation="softmax"))
        # model.compile(optimizer=rmsprop, loss="categorical_crossentropy", metrics=["accuracy"])
        lstm_layer = LSTM(config['lstm']['num_units'], recurrent_dropout=config['lstm']['dropout'],
                          return_sequences=True)
        lstm_layer1 = LSTM(config['lstm']['num_units'], recurrent_dropout=config['lstm']['dropout'])
        dropout_layer = Dropout(config['dropout']['val'])
        dense_layer = Dense(1, activation="sigmoid")
        # model.add(Bidirectional(LSTM(config['lstm']['num_units'],
        #                     recurrent_dropout=config['lstm']['dropout'])))
        model.add(lstm_layer)
        model.add(lstm_layer1)
        model.add(dropout_layer)
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))
        model.add(Dropout(0.3, noise_shape=None, seed=None))
        model.add(Dense(50, activation="relu"))
        model.add(Dropout(0.2, noise_shape=None, seed=None))
        model.add(Dense(10, activation="relu"))
        # # model.add(Dense(2, activation="softmax"))
        model.add(dense_layer)
        opt = Adam(learning_rate=0.01)
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        return model
