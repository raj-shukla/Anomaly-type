import numpy as np
import random
from scipy import stats
from scipy.stats import kurtosis, skew
import process_data
import math
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM, Dropout
import csv
import pandas as pd

np.set_printoptions(threshold=np.nan)
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

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train, Y_train, batch_size= 256, verbose=1, epochs=5000)

model.save('model_deep_1000.h5')
model.summary()

with open('model_architecture_10000.json', 'w') as f:
    f.write(model.to_json())


prediction_train = model.predict(X_train).flatten()
error_train = np.mean(np.abs(prediction_train - Y_train))

prediction_test = model.predict(X_test).flatten()
error_test = np.mean(np.abs(prediction_test - Y_test))

print(prediction_train)
print(prediction_test)
print(error_train)
print(error_test)




   
        
