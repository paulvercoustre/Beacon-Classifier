import os
import numpy as np
import pandas as pd

import data_manager as dm

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import RNN, SimpleRNN, Dense, Activation, Dropout

FLAGS = None
ROOT = os.path.join(os.path.dirname(__file__), '../')

file_name = os.path.join(ROOT, 'data', 'no_filter.csv')
pickle_file = os.path.join(ROOT, 'cache', 'telemetry.npy')


def main():

    # import the data
    # data = np.genfromtxt(fname=file_name,
    #                      delimiter=",",
    #                      skip_header=1,
    #                      usecols=range(1, 24),
    #                      filling_values=-120)
    #
    # # create features and labels
    # X = data[:, 2:24]  # each feature is the signal intensity of a given beacon at a given time step
    # y = data[:, 0]
    #
    # n_samples, n_features = X.shape
    #
    # X = X.reshape(n_samples, 7, 3)
    # X = X.transpose(0, 2, 1)

    extractor = dm.FeatureLabelExtractor(pickle_file)
    X = extractor.get_sequential_features
    y = extractor.get_labels

    # pad the telemetry to 4 timesteps, fill with -120 empty data
    X = pad_dataframe(X, 4, -120)

    n_samples, timesteps, m_features = X.shape

    print('\nBuilding Model...')
    model = Sequential()
    model.add(SimpleRNN(8, return_sequences=True, input_shape=(timesteps, m_features)))
    model.add(Dropout(0.9))
    model.add(SimpleRNN(8))
    model.add(Dense(8))
    model.add(Activation('softmax'))
    print('Model Built')

    print('\nCompiling Model...')
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())


def pad_dataframe(dataframe, maxlen, value):

    padded = []
    for col in dataframe.columns.values:
        padded.append(pad_sequences(sequences=dataframe[col],
                                    maxlen=maxlen,
                                    padding='pre',
                                    truncating='pre',
                                    value=value))

    padded = np.asarray(padded).transpose(1, 2, 0)
    return padded

if __name__ == '__main__':
    main()
