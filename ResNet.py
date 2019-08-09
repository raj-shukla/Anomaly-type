from tensorflow import keras
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

np.random.seed(813306)
 
def build_resnet(input_shape, n_feature_maps):
    print ('build conv_x')
    x = keras.layers.Input(shape=(input_shape))
    conv_x = keras.layers.BatchNormalization()(x)
    conv_x = keras.layers.Conv2D(n_feature_maps, 8, 1, padding='same')(conv_x)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
     
    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps, 1, 1,padding='same')(x)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x)
    print ('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)
     
    print ('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv2D(n_feature_maps*2, 8, 1, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps*2, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps*2, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
     
    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps*2, 1, 1,padding='same')(x1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x1)
    print ('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)
     
    print ('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv2D(n_feature_maps*2, 8, 1, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps*2, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps*2, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps*2, 1, 1,padding='same')(x1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x1)
    print ('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)
     
    full = keras.layers.GlobalAveragePooling2D()(y)
    out = keras.layers.Dense(units=1)(full)
    print ('        -- model was built.')
    return x, out
 
       
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

X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))

Y = np.reshape(Y, (X.shape[0], 1))

X_train = X[0:2000, :, :, :]
Y_train = Y[0:2000]

X_test = X[2000:, :, :, :]
Y_test = Y[2000:]

      
layers = 4
epochs = 1500
file_name = "ResNet_layers_"+ str(layers)+ "_" + "epochs_" + str(epochs)
     
x , y = build_resnet(X_train.shape[1:], 64)
model = keras.models.Model(inputs=x, outputs=y)
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mae', 'acc'])
#reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001) 

history = model.fit(X_train, Y_train, batch_size= 256, verbose=1, epochs=epochs)



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


