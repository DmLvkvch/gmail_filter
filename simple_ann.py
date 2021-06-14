import pandas as pd
from keras.layers import Dense, Dropout, LSTM, Embedding, Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, classification_report, roc_auc_score, precision_score, accuracy_score, \
    recall_score, f1_score, balanced_accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
import re

def create_baseline():
    # create model
    model = Sequential()
    lstm_layer = LSTM(56, input_shape=(57, 1), return_sequences=True, recurrent_dropout=0.2)
    lstm_layer1 = LSTM(128, return_sequences=False, recurrent_dropout=0.2)
    # model.add(Dense(57, input_dim=57, activation='relu'))
    # model.add(Dense(25, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(lstm_layer)
    model.add(Flatten())
    model.add(Dense(50, activation = "relu"))
    model.add(Dropout(0.3, noise_shape=None, seed=None))
    model.add(Dense(50, activation = "relu"))
    model.add(Dropout(0.2, noise_shape=None, seed=None))
    model.add(Dense(10, activation = "relu"))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


dataset_path = '../data/collections/spambase.data'

spamdata = pd.read_csv(dataset_path, sep=",", header=None)



# Set attribute names
attributeNames = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
                  'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
                  'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses',
                  'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
                  'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp',
                  'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs',
                  'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
                  'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
                  'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
                  'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(',
                  'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average',
                  'capital_run_length_longest', 'capital_run_length_total', 'spam_or_not']
spamdata.columns = attributeNames
dataset = spamdata.values


def load_data_from_file(path: str) -> tuple:
    file = open(path, 'r', encoding="utf8")
    Lines = file.readlines()
    labels = []
    emails = []
    for line in Lines:
        label, email = line.strip().split(maxsplit=1)
        labels.append(label.strip())
        emails.append(email.strip())
    return labels, emails


attr = []
for i in range(len(attributeNames)):
    if(i<48):
        attr.append(attributeNames[i][10: len(attributeNames[i])])

labels, emails = load_data_from_file('../data/collections/SMSSpamCollection.txt')
Y = []
for l in labels:
    if l == 'ham':
        Y.append(1)
    else:
        Y.append(0)

X = []
# attr.append(';')
# attr.append('(')
# attr.append('[')
# attr.append('!')
# attr.append('$')
# attr.append('#')

def get_max_uppercase_run_from_string(s):
    # construct a list of all the uppercase segments in your string
    list_of_uppercase_runs = re.findall(r"[A-Z]+", s)

    # find out what the longest string is in your list
    if len(list_of_uppercase_runs)==0:
        return 0, 0, 0
    longest_string = max(list_of_uppercase_runs, key=len)
    tmp = []
    for l in list_of_uppercase_runs:
        tmp.append(len(l))

    # return the length of this string to the user
    return sum(tmp)/len(tmp), len(longest_string), len(list_of_uppercase_runs)


for e in emails:
    tmp = []
    for a in attr:
        count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(a), e))
        tmp.append(100*count/len(e.split()))
    count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(';'), e))
    tmp.append(100*count/len(e))
    count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('('), e))
    tmp.append(100*count/len(e))
    count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('['), e))
    tmp.append(100*count/len(e))
    count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('!'), e))
    tmp.append(100*count/len(e))
    count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('$'), e))
    tmp.append(100*count/len(e))
    count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('#'), e))
    tmp.append(100*count/len(e))
    a, b, c = get_max_uppercase_run_from_string(e)
    tmp.append(a)
    tmp.append(b)
    tmp.append(c)
    X.append(tmp)


# split into input (X) and output (Y) variables
# Y = np.array(Y)
X = dataset[:, 0:57]
Y = dataset[:, -1]
X = np.reshape(X, (len(X), 57, 1))
print(X.shape)
print(Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# estimator = KerasClassifier(build_fn=create_baseline, epochs=30, batch_size=5, verbose=1)
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# cvscores = []
#
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print(results)
# print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
epochs = 40
epochs_array = range(1, epochs + 1)
# create model
model = create_baseline()
# Fit the model
kfold = KFold(n_splits=10, shuffle=True)
fold_no = 1
acc_per_fold = []
loss_per_fold = []
prec_per_fold = []


def save_file(file, val):
    f = open('res/res/'+file, "a")
    f.write("% f \n" % val)
    f.close()

class Metrics(Callback):
    def __init__(self, x, y):
        self.val_x = x
        self.val_y = y

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_ras = []
        self.val_ba = []
        save_file('prec2.txt', 0.0)
        save_file('recall2.txt', 0.0)
        save_file('f12.txt', 0.0)
        save_file('ras2.txt', 0.0)
        save_file('acc2.txt', 0.0)
        save_file('ba2.txt', 0.0)

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
        _val_ba = balanced_accuracy_score(val_targ, val_predict)
        self.val_ba.append(_val_ba)
        print ("% f, % f, % f, % f, % f %f\n" % (_val_precision, _val_recall, _val_f1, _val_ras, _val_acc, _val_ba))
        save_file('prec2.txt', _val_precision)
        save_file('recall2.txt', _val_recall)
        save_file('f12.txt', _val_f1)
        save_file('ras2.txt', _val_ras)
        save_file('acc2.txt', _val_acc)
        save_file('ba2.txt', _val_ba)
        return


for tr, test in kfold.split(X, Y):
    seq_model = create_baseline()

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    seq_model.fit( X[tr], Y[tr], validation_data=(X[test], Y[test]), epochs=50, verbose=1, callbacks=[Metrics(X[test], Y[test])])
    # Generate generalization metrics
    scores = seq_model.evaluate(X[test], Y[test], verbose=0)
    print(f'Score for fold {fold_no}: {seq_model.metrics_names[0]} of {scores[0]}; {seq_model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    prec_per_fold.append(precision_score(Y[test], seq_model.predict(X[test]).round()))
    # Increase fold number
    labels_test = Y[test]
    texts_test = X[test]
    fold_no = fold_no + 1
    print("PRECISION:")
    precision = precision_score(labels_test, seq_model.predict(texts_test).round())
    print(precision)
    print("ACCURACY:")
    accuracy = accuracy_score(labels_test, seq_model.predict(texts_test).round())
    print(accuracy)
    f = open("acc_prec.txt", "a")
    f.write("% f, % f \n" % (precision, accuracy))
    f.close()
    rcs = metrics.recall_score(labels_test, seq_model.predict(texts_test).round())
    f1 = metrics.f1_score(labels_test, seq_model.predict(texts_test).round())
    cp = classification_report(labels_test, seq_model.predict(texts_test).round())
    print('rcs ', rcs)
    print('cp', cp)
    ras = roc_auc_score(labels_test, seq_model.predict(texts_test).round())
    print(ras)


print(acc_per_fold)
print(prec_per_fold)
