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



model = load_model('model_deep_1000.h5')
#model.summary()
model.get_weights()

prediction_test = model.predict(X_test)
print (prediction_test.flatten())

error = 0
for i in range(0, len(Y_test)):
    print("###############")
    #print(Y_test[i])
    #print(prediction_test[i])
    diff = abs(prediction_test[i] - Y_test[i])
    print(diff)
    error = error + diff
    
print(error/len(Y_test))









