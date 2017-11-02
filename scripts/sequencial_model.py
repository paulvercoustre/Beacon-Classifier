import os
import numpy as np
import pandas as pd
import random as rn

import data_manager as dm
import utils

from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import RNN, SimpleRNN, Dense, Activation, Dropout

import tensorflow as tf

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
seed = 0

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

FLAGS = None
ROOT = os.path.join(os.path.dirname(__file__), '../')


csv_file = os.path.join(ROOT, 'data', 'no_filter.csv')
pickle_file = os.path.join(ROOT, 'cache', 'telemetry.npy')


def main():

    if True:
        extractor = dm.CsvExtractor(csv_file)
        X = extractor.get_sequential_features
        y = extractor.get_labels

    if False:
        extractor = dm.PickleExtractor(pickle_file)
        X = extractor.get_sequential_features
        y = extractor.get_labels

        # pad the telemetry to 4 timesteps, fill with -120 empty values
        X = utils.pad_dataframe(X, 4, -120)

    y = np_utils.to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test)

    n_samples, timesteps, m_features = X.shape
    batch_size = 32
    epochs = 30

    print('\nBuilding Model...')
    model = Sequential()
    model.add(SimpleRNN(8, return_sequences=True, input_shape=(timesteps, m_features)))
    # model.add(Dropout(0.9))
    model.add(SimpleRNN(8))
    model.add(Dense(8))
    model.add(Activation('softmax'))
    print('Model Built')

    print('\nCompiling Model...')
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    print('\nTraining Model...')
    model.fit(x=X_train, y=y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
              verbose=1)

    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)


if __name__ == '__main__':
    main()
