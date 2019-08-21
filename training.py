import numpy as np
import itertools
import matplotlib.pyplot as plt
import csv
import os
import process_data

pollutants = process_data.pollutants
WINSPD = process_data.WINSPD
TEMP = process_data.TEMP
traffic = process_data.traffic

pollutants_name = ["CO", "BC", "NO2", "NOX", "PM25HR"]

def write_data(pollutant, pollutant_name):
    l = 16
    X = []
    Y = []
    for i in range(0, len(TEMP) - l):
        x = []
        x.extend(pollutant[i:i+l])
        x.extend(TEMP[i:i+l])
        x.extend(WINSPD[i:i+l])
        y = traffic[i+l-1]
        X.append(x)
        Y.append(y)
        
   
        
    with open("training_data/" + pollutant_name + "_input_data.csv", mode="w") as f:
        writer = csv.writer(f)
        for row in X:
            writer.writerow(row)
        
    with open("training_data/" + pollutant_name +"_output_data.csv", mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(Y)
            
    

for i in range(0, len(pollutants)):
    write_data(pollutants[i], pollutants_name[i])
    
#os.system("./run.sh")

    
    
