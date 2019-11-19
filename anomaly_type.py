from tensorflow import keras
import numpy as np
import itertools
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import Dense, Flatten
from keras.layers import LSTM, Dropout
import random
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
    

def add_outliers(Y_test, positions,  magnitude):
    for i in positions:
        Y_test[i] = Y_test[i] + max(magnitude, 1-Y_test[i])
    return Y_test
    
def find_accuracy(positions, outliers_positions):
    all_positions = np.asarray(range(0, 952))
    not_positions = np.setdiff1d(all_positions, positions)
    tp = np.sum([1 if outliers_positions[i] == 1 else 0 for i in positions])
    fp = len(positions) - tp
    fn = np.sum([1 if outliers_positions[i] == 1 else 0 for i in not_positions])
    
    P = round(float(tp/(tp + fp)), 2)
    R = round(float(tp/(tp + fn)), 2)
    F = round(float((2*P*R)/(P+R)), 2)
    
    return (tp, fp, fn, P, R, F)   

th = 0.05
std_dev = 0.31
#positions = random.sample(range(0, 952), 476)
alpha_list = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5] 
beta_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
K_list = [1, 2, 3, 4, 5]

def run_simulation(alpha, beta, K, positions):
    outliers = []
    for arg in pollutants_name:
        X_test, Y_test = read_data(arg)
        Y_predict = predict_model(X_test, arg)
        Y_test = add_outliers(Y_test, positions, alpha*std_dev)
        diff = np.abs(Y_test - Y_predict)
        outlier = np.array([1 if i > beta*th else 0 for i in diff])
        outliers.append(outlier)

    outliers = np.array(outliers)
    counts = np.sum(outliers, axis=0)

    outliers_position = [1 if i>=K else 0 for i in counts]
    accuracy = find_accuracy(positions, outliers_position)
    return accuracy

'''    
for alpha in alpha_list:
    accuracy = run_simulation(alpha, beta_list[2], K_list[2])
    print (accuracy[3:])

for beta in beta_list:
    accuracy = run_simulation(alpha_list[2], beta, K_list[2])
    print (accuracy[3:])

for K in K_list:
    accuracy = run_simulation(alpha_list[2], beta_list[4], K)
    print (accuracy[3:])
    
'''    

'''
accuracy_list = []    
for alpha in alpha_list:
    for beta in beta_list:
        for K in K_list:
            accuracy = run_simulation(alpha, beta, K)
            accuracy_list.append((alpha, beta, K) + accuracy)
            print ((alpha, beta, K) + accuracy)
         
print (accuracy_list)

with open("accuracy.txt", mode="w") as f:
    writer = csv.writer(f)
    for row in accuracy_list:
        writer.writerow(row)

'''

accuracy_list = []
f = 0
for i in range(0, 20):
    m = int(952/20)
    f = f + m
    positions = random.sample(range(0, 952), f)
    accuracy = run_simulation(alpha_list[2], beta_list[4], K_list[2], positions)
    accuracy_list.append(accuracy)
    print (accuracy)
    
print (accuracy)        







