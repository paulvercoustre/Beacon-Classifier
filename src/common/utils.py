import numpy as np
from keras.preprocessing.sequence import pad_sequences


def pad_dataframe(dataframe, maxlen, value):

    # to get the exact same data as initial extract, choose from this list
    beacons = ['F74B56F7-5F12-413B-BF5E-DF09CC7E5C33_1_1',
               'F74B56F7-5F12-413B-BF5E-DF09CC7E5C33_1_2',
               'F74B56F7-5F12-413B-BF5E-DF09CC7E5C33_2_1',
               'F74B56F7-5F12-413B-BF5E-DF09CC7E5C33_2_2',
               'F74B56F7-5F12-413B-BF5E-DF09CC7E5C33_3_1',
               'F74B56F7-5F12-413B-BF5E-DF09CC7E5C33_3_2',
               'F74B56F7-5F12-413B-BF5E-DF09CC7E5C33_3_3']

    padded = []
    for col in beacons:
        padded.append(pad_sequences(sequences=dataframe[col],
                                    maxlen=maxlen,
                                    padding='pre',
                                    truncating='pre',
                                    value=value))

    padded = np.asarray(padded).transpose(1, 2, 0)
    return padded
