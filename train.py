import keras.callbacks
import numpy as np
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, \
    accuracy_score


class Metrics(Callback):
    def __init__(self, x, y):
        self.val_x = x
        self.val_y = y

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_ras = []

    def on_epoch_end(self, epoch, logs={}):
        tmp = self.model.predict(self.val_x)
        val_predict = (np.asarray(tmp)).round()
        val_targ = self.val_y
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_f1 = f1_score(val_targ, val_predict)
        _val_ras = roc_auc_score(val_targ, val_predict)
        _val_acc = accuracy_score(val_targ, val_predict)
        self.val_ras.append(_val_ras)
        self.val_recalls.append(_val_recall)
        self.val_f1s.append(_val_f1)
        self.val_precisions.append(_val_precision)
        print("% f, % f, % f, % f, % f \n" % (_val_precision, _val_recall, _val_f1, _val_ras, _val_acc))
        return


class TrainModel:

    def train_model(self, seq_model, config: dict, X_train, y_train, X_test, y_test):
        checkpoint_file_path = "../checkpoints/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(
            checkpoint_file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        callbacks = []
        callbacks.append(checkpoint)
        csv_logger = CSVLogger('training.log')
        callbacks.append(csv_logger)
        callbacks.append(keras.callbacks.TensorBoard(log_dir="."))
        metrics_binary = Metrics(X_test, y_test)
        # estimator = KerasClassifier(build_fn=seq_model, epochs=config['train']['epochs'],
        #                             batch_size=config['train']['batch_size'],
        #                             callbacks=callbacks, verbose=1)
        # kfold = StratifiedKFold(n_splits=10, shuffle=True)
        # cvscores = []
        # # evaluate model with standardized dataset
        #
        # results = cross_val_score(estimator, X, Y[:, 0], cv=kfold)
        # print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
        callbacks.append(metrics_binary)
        results = seq_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                batch_size=config['train']['batch_size'],
                                epochs=config['train']['epochs'],
                                callbacks=callbacks,
                                verbose=1)

        # hist = results.history
        # print(hist)
        # plt.title('Training and test accuracy ')
        # plt.plot(hist['accuracy'], 'r', label='train')
        # plt.plot(hist['val_accuracy'], 'b', label='test')
        # plt.legend()
        # plt.show()
