from tensorflow import keras
import numpy as np
import random
from scipy import stats
from scipy.stats import kurtosis, skew
import math
from keras.callbacks import History, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM, Dropout
from keras import metrics
import json
import csv

np.random.seed(7)
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

X, Y = np.array(X), np.array(Y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
Y = np.reshape(Y, (X.shape[0], 1))
X_train = X[0:2000, :, :]
Y_train = Y[0:2000]
X_test = X[2000:, :, :]
Y_test = Y[2000:]
    

layers = 4
epochs = 1000

model = Sequential ()
model.add(LSTM(units=128, return_sequences=True, activation='softmax', input_shape=(X_train.shape[1], 1))) 
model.add(Dropout(0.2)) 
for i in range(0, layers-2): 
    model.add(LSTM(units=50, activation='softmax', return_sequences=True))  
    model.add(Dropout(0.2))

model.add(LSTM(units=50, activation='softmax'))  
model.add(Dropout(0.2))  
model.add(Dense(units = 1))  
#model.add(Flatten())



model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mae', 'acc'])

tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(X_train, Y_train, batch_size= 256, verbose=1, epochs=epochs,
                  callbacks=[tensor_board])
#print(file_name)

model.save(file_name + '.h5')
model.summary()

score = model.evaluate(X_test, Y_test)
print (score)

        
