import os
import numpy as np
import pandas as pd
import random as rn
import matplotlib.pyplot as plt

import data_manager as dm
import utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import RNN, SimpleRNN, LSTM, GRU, Dense, Activation, Dropout
from keras import optimizers
from keras import regularizers

import tensorflow as tf

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
seed = 1

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

FLAGS = None
ROOT = os.path.join(os.path.dirname(__file__), '../')


csv_file = os.path.join(ROOT, 'data', 'no_filter.csv')
pickle_file = os.path.join(ROOT, 'cache', 'telemetry_new.npy')


def main():

    if False:
        extractor = dm.CsvExtractor(csv_file)
        X = extractor.get_sequential_features
        y = extractor.get_labels

    if True:
        extractor = dm.PickleExtractor(pickle_file)
        X = extractor.get_sequential_features
        y = extractor.get_labels

        # pad the telemetry to 4 timesteps, fill with -120 empty values
        X = utils.pad_dataframe(X, 3, -120)

    # Transform labels to One-Hot encoded vectors
    y = np_utils.to_categorical(y)

    # split in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    n_samples, timesteps, m_features = X_train.shape

    # Standardise features
    X_train = X_train.reshape(-1, timesteps*m_features)
    X_test = X_test.reshape(-1, timesteps*m_features)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).reshape(-1, timesteps, m_features)
    X_test = scaler.transform(X_test).reshape(-1, timesteps, m_features)

    print(X_train.shape)
    print(X_test.shape)

    print('\nDataset shapes:')
    print('* %s Training & Validation Samples' % X_train.shape[0])
    print('* %s Testing Samples' % X_test.shape[0])

    batch_size = 32
    epochs = 30

    print('\nBuilding Model...')
    model = Sequential()
    model.add(LSTM(64, input_shape=(timesteps, m_features),
                   return_sequences=False,
                   # dropout=0.1,
                   # recurrent_dropout=0.0,
                   kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(8))
    model.add(Activation('softmax'))
    print('Model Built')

    # choose the best optimiser
    adam = optimizers.Adam(lr=0.0005)
    sgd = optimizers.sgd(lr=0.0005, momentum=0.01)
    rmsprop = optimizers.rmsprop(lr=0.0005)

    print('\nCompiling Model...')
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    print('\nTraining Model...')
    history = model.fit(x=X_train, y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2,
                        verbose=1)

    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Learning Curve")
    #plt.show()


if __name__ == '__main__':
    main()
