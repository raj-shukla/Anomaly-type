import numpy as np
import random
from scipy import stats
from scipy.stats import kurtosis, skew
import math
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import Dense, Flatten
from keras.layers import LSTM, Dropout
import csv

np.random.seed(7)


#X = process_data.X
#Y = process_data.Y
X = []
Y = []
with open("input_data.csv", "r") as input_file:
    reader = csv.reader(input_file)
    for row in reader:
        x = [float(i) for i in row]
        X.append(x)
        
with open("output_data.csv", "r") as output_file:
    reader = csv.reader(output_file)
    for row in reader:
        Y = [float(i) for i in row]

print(X[0:10])


X, Y = np.array(X), np.array(Y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

Y = np.reshape(Y, (X.shape[0], 1))

X_train = X[0:2000, :, :]
Y_train = Y[0:2000]

X_test = X[2000:, :, :]
Y_test = Y[2000:]

model = load_model('MLP.h5')
model.summary()
model.get_weights()

prediction_test = model.predict(X_test)
print (prediction_test.flatten())

model = load_model('LSTM.h5')
model.summary()
model.get_weights()

prediction_test = model.predict(X_test)
print (prediction_test.flatten())

model = load_model('CNN.h5')
model.summary()
model.get_weights()

prediction_test = model.predict(X_test)
print (prediction_test.flatten())

model = load_model('CNN_LSTM.h5')
model.summary()
model.get_weights()

prediction_test = model.predict(X_test)
print (prediction_test.flatten())

model = load_model('ResNet.h5')
model.summary()
model.get_weights()

prediction_test = model.predict(X_test)
print (prediction_test.flatten())








