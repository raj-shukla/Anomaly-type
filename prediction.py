import numpy as np
import random
from scipy import stats
from scipy.stats import kurtosis, skew
import process_data
import math
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import Dense, Flatten
from keras.layers import LSTM, Dropout
import csv
import pandas as pd

np.set_printoptions(threshold=np.nan)

np.random.seed(7)


X = process_data.X
Y = process_data.Y

X, Y = np.array(X), np.array(Y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

Y = np.reshape(Y, (X.shape[0], 1))

X_train = X[0:2000, :, :]
Y_train = Y[0:2000]

X_test = X[2000:, :, :]
Y_test = Y[2000:]



model = load_model('model_10000.h5')
#model.summary()
model.get_weights()

prediction_train = model.predict(X_train).flatten()
error_train = np.mean(np.abs(prediction_train - Y_train))

prediction_test = model.predict(X_test).flatten()
error_test = np.mean((np.abs(prediction_test - Y_test))

print(prediction_train)
print(prediction_test)
print(error_train)
print(error_test)

print(prediction_test[10:20])
print(Y_test[10:20])










