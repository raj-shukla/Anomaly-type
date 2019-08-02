import numpy as np
import random
from scipy import stats
from scipy.stats import kurtosis, skew
import math
from keras.callbacks import History
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM, Dropout
from keras import metrics
import json
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



X, Y = np.array(X), np.array(Y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

Y = np.reshape(Y, (X.shape[0], 1))

X_train = X[0:2000, :, :]
Y_train = Y[0:2000]

X_test = X[2000:, :, :]
Y_test = Y[2000:]





print(X.shape)
print(Y.shape)       



model = Sequential ()
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1))) 
model.add(Dropout(0.2))  
model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=50))  
model.add(Dropout(0.2))  
model.add(Dense(units = 1))  
#model.add(Flatten())

layers = 8
epochs = 2

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mae', 'acc'])

history = model.fit(X_train, Y_train, batch_size= 256, verbose=1, epochs=epochs)
file_name = "model_layers_"+ str(layers)+ "_" + "epochs_" + str(epochs)

print(file_name)

model.save(file_name + '.h5')
model.summary()

with open(file_name +  '.json', 'w') as f:
    f.write(model.to_json())
    
with open( file_name + '_history.json', 'w') as f:
    json.dump(history.history, f)


prediction_train = model.predict(X_train).flatten()
error_train = np.mean(np.abs(prediction_train - Y_train))

prediction_test = model.predict(X_test).flatten()
error_test = np.mean(np.abs(prediction_test - Y_test))





   
        
