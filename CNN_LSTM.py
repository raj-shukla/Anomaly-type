from tensorflow import keras
import numpy as np
import random
from keras.callbacks import History, TensorBoard, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
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
epochs = 1500
file_name = "CNN_LSTM_layers_"+ "activation_sigmoid_" + str(layers)+ "_" + "epochs_" + str(epochs)

model = Sequential ()
model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.5))
model.add(LSTM(units=50, activation='softmax', return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=50, activation='softmax'))  
model.add(Dropout(0.2))  
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics=['mae', 'mse'])

tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
reduce_lr = ReduceLROnPlateau(monitor='mean_absolute_error', factor=0.5, patience=50, min_lr=0.0001)
history = model.fit(X_train, Y_train, batch_size= 256, verbose=1, epochs=epochs, validation_data=(X_test, Y_test), 
                  callbacks=[tensor_board, reduce_lr])

print(file_name)
#model.save(file_name + '.h5')
model.summary()

score = model.evaluate(X_test, Y_test)
print (score)


   
        
