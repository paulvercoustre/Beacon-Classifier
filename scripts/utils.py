import numpy as np
from keras.preprocessing.sequence import pad_sequences


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
