from tensorflow import keras
import numpy as np
import itertools
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import Dense, Flatten
from keras.layers import LSTM, Dropout
import csv
import functions


pollutants_name = ["CO", "BC", "NO2", "NOX", "PM25HR"]

def read_data(arg):
    X, Y = functions.read_data(arg)
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1]))
    Y = np.reshape(Y, (X.shape[0]))
    X_test = X[2000:, :]
    Y_test = Y[2000:]  
    
    return X_test, Y_test


def predict_model (X_test, arg):
    file_name = arg + "_MLP"
    model = load_model("models/" + file_name + '.h5')
    model.get_weights()
    predictions = model.predict(X_test)
    
    return predictions.flatten()
    

def add_outliers(Y_test, magnitude):
    #Y_test[200:300] = np.array([i + magnitude for i in Y_test[200:300]])
    return Y_test
    
arg = "CO"
X_test, Y_test = read_data(arg)
Y_predict = predict_model(X_test, arg)
Y_test = add_outliers(Y_test, 0.1)

diff = np.abs(Y_predict - Y_test)

print ([1 if i>0.13 else 0 for i in diff[100:200]])
print (Y_predict [100:200])
print (Y_test[100:200])

print (sum([1 if i>0.13 else 0 for i in diff[100:200]]))
print (sum([1 if i>0.20 else 0 for i in diff[100:200]]))
print (sum([1 if i>0.35 else 0 for i in diff[100:200]]))
print (sum([1 if i>0.40 else 0 for i in diff[100:200]]))
print (np.mean(diff))
print (np.max(diff))
print (np.min(diff))

Y_predict = [i-0.2 if i<0.2 else i for i in Y_predict]
Y_predict = [i-0.2 if i>0.7 else i for i in Y_predict]
a = list(range(952))
plt.plot (a, Y_predict, a, Y_test)
plt.show()




