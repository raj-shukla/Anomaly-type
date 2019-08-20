from tensorflow import keras
import numpy as np
import random
from matplotlib import pyplot as plt
from keras.callbacks import History, TensorBoard, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import metrics
import json
import csv
import sys

def read_data(arg):
    np.random.seed(7)
    X = []
    Y = []

    with open("training_data/" + arg + "_input_data.csv", "r") as input_file:
        reader = csv.reader(input_file)
        for row in reader:
            x = [float(i) for i in row]
            X.append(x)
        
    with open("training_data/" + arg + "_output_data.csv", "r") as output_file:
        reader = csv.reader(output_file)
        for row in reader:
            Y = [float(i) for i in row]
     
    return X, Y
     
epochs = 1

def write_model(file_name, model, history):
    model.save("models/" + file_name + '.h5')
    model.summary()


    with open("history/" + file_name + '.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in history.history.items():
            row = [key]
            row.extend(value) 
            writer.writerow(row)



