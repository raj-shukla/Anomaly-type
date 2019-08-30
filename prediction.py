from tensorflow import keras
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
import functions

pollutants_name = ["CO", "BC", "NO2", "NOX", "PM25HR"]



def predict(X_train, Y_train, X_test, Y_test, file_name):
    model = load_model("models/" + file_name + '.h5')
    model.get_weights()
    scores = model.evaluate(X_test, Y_test)
    print ((round(scores[1], 2),  round(scores[2], 2)))




def MLP(i):
    for j in range(0, 1):
        X, Y = functions.read_data(i)
        X, Y = np.array(X), np.array(Y)
        X = np.reshape(X, (X.shape[0], X.shape[1]))
        Y = np.reshape(Y, (X.shape[0]))
        X_train = X[0:2000, :]
        Y_train = Y[0:2000]
        X_test = X[2000:, :]
        Y_test = Y[2000:]
        file_name = i + "_MLP"
        predict (X_train, Y_train, X_test, Y_test, file_name)
        
def LSTM(i):
    for j in range(0, 1):
        X, Y = functions.read_data(i)
        X, Y = np.array(X), np.array(Y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        Y = np.reshape(Y, (X.shape[0], 1))
        X_train = X[0:2000, :, :]
        Y_train = Y[0:2000]
        X_test = X[2000:, :, :]
        Y_test = Y[2000:]
        file_name = i + "_LSTM"
        predict (X_train, Y_train, X_test, Y_test, file_name)
        
def CNN(i):
    for j in range(0, 1):
        X, Y = functions.read_data(i)
        X, Y = np.array(X), np.array(Y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        Y = np.reshape(Y, (X.shape[0], 1))
        X_train = X[0:2000, :, :]
        Y_train = Y[0:2000]
        X_test = X[2000:, :, :]
        Y_test = Y[2000:]   
        file_name = i + "_CNN"
        predict (X_train, Y_train, X_test, Y_test, file_name)


def CNN_LSTM(i):
    for j in range(0, 1):
        X, Y = functions.read_data(i)
        X, Y = np.array(X), np.array(Y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        Y = np.reshape(Y, (X.shape[0], 1))
        X_train = X[0:2000, :, :]
        Y_train = Y[0:2000]
        X_test = X[2000:, :, :]
        Y_test = Y[2000:]
        file_name = i + "_CNN_LSTM"
        predict (X_train, Y_train, X_test, Y_test, file_name)
        
def ResNet(i):
    for j in range(0, 1):
        X, Y = functions.read_data(i)
        X, Y = np.array(X), np.array(Y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
        Y = np.reshape(Y, (X.shape[0], 1))
        X_train = X[0:2000, :, :, :]
        Y_train = Y[0:2000]
        X_test = X[2000:, :, :, :]
        Y_test = Y[2000:]
        file_name = i + "_ResNet"
        predict (X_train, Y_train, X_test, Y_test, file_name)



for i in pollutants_name:
    MLP(i)
    LSTM(i)
    CNN(i)
    CNN_LSTM(i)
    #ResNet(i)








