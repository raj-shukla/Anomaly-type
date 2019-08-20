from tensorflow import keras
import numpy as np
import random
from keras.callbacks import History, TensorBoard, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout
from keras import metrics
import functions
import csv
import sys

arg = sys.argv[1]
X, Y = functions.read_data(arg)
file_name = arg + "_MLP"
epochs = functions.epochs

X, Y = np.array(X), np.array(Y)
X = np.reshape(X, (X.shape[0], X.shape[1]))
Y = np.reshape(Y, (X.shape[0]))
X_train = X[0:2000, :]
Y_train = Y[0:2000]
X_test = X[2000:, :]
Y_test = Y[2000:]
    
    
model = Sequential ()
model.add(Dense(72, activation = 'relu', input_shape=(X_train.shape[1], )))
model.add(Dropout(0.4)) 
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units = 1))  
#model.add(Flatten())



model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mae', 'mse'])

tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
reduce_lr = ReduceLROnPlateau(monitor='mean_absolute_error', factor=0.5, patience=50, min_lr=0.0001)
history = model.fit(X_train, Y_train, batch_size= 256, verbose=1, epochs=epochs, validation_data=(X_test, Y_test), 
                  callbacks=[tensor_board, reduce_lr])


print(file_name)
functions.write_model(file_name, model, history)


