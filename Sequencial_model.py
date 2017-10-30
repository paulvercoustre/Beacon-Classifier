import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, RNN

data = np.genfromtxt('no_filter.csv', delimiter=",", skip_header=1, usecols=range(1, 24), filling_values=-120)

# create features and labels
X = data[:, 2:24]  # each feature is the signal intensity of a given beacon at a given time step
y = data[:, 0]

n_samples, n_features = X.shape

X = X.reshape(n_samples, 7, 3)
X = X.transpose(0, 2, 1)
print(X.shape)

model = Sequential()
model.add(RNN(10, ))


