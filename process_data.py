import numpy as np
import itertools
import matplotlib.pyplot as plt
import csv
import data


CO_h = np.asarray(data.CO_h)
BC_h = np.asarray(data.BC_h)
NO2_h = np.asarray(data.NO2_h)
NOX_h = np.asarray(data.NOX_h)
PM25HR_h = np.asarray(data.PM25HR_h)
TEMP_h = np.asarray(data.TEMP_h)
WINSPD_h = np.asarray(data.WINSPD_h)

traffic_m = np.asarray(data.traffic_m)

CO_h = (CO_h - min(CO_h))/(max(CO_h) - min(CO_h)) 
BC_h = (BC_h - min(BC_h))/(max(BC_h) - min(BC_h)) 
NO2_h = (NO2_h - min(NO2_h))/(max(NO2_h) - min(NO2_h)) 
NOX_h = (NOX_h - min(NOX_h))/(max(NOX_h) - min(NOX_h)) 
PM25HR_h = (PM25HR_h - min(PM25HR_h))/(max(PM25HR_h) - min(PM25HR_h)) 
TEMP_h = (TEMP_h - min(TEMP_h))/(max(TEMP_h) - min(TEMP_h))
WINSPD_h = (WINSPD_h - min(WINSPD_h))/(max(WINSPD_h) - min(WINSPD_h))  

traffic_m = (traffic_m - min(traffic_m))/(max(traffic_m) - min(traffic_m)) 

print(len(traffic_m))

CO = list(itertools.chain.from_iterable(itertools.repeat(x, 4) for x in CO_h))
BC = list(itertools.chain.from_iterable(itertools.repeat(x, 4) for x in BC_h))
NO2 = list(itertools.chain.from_iterable(itertools.repeat(x, 4) for x in NO2_h))
NOX = list(itertools.chain.from_iterable(itertools.repeat(x, 4) for x in NOX_h))
PM25HR = list(itertools.chain.from_iterable(itertools.repeat(x, 4) for x in PM25HR_h))
TEMP = list(itertools.chain.from_iterable(itertools.repeat(x, 4) for x in TEMP_h))
WINSPD = list(itertools.chain.from_iterable(itertools.repeat(x, 4) for x in WINSPD_h))

traffic = [ sum(traffic_m[i:i+3]) for i in range(0, len(traffic_m), 3)]
traffic = (traffic - min(traffic))/(max(traffic) - min(traffic))
 

l = 4
X = []
Y = []

for i in range(0, len(CO) - l):
    x = []
    x.extend(CO[i:i+l])
    x.extend(BC[i:i+l])
    x.extend(NO2[i:i+l])
    x.extend(NOX[i:i+l])
    x.extend(PM25HR[i:i+l])
    x.extend(TEMP[i:i+l])
    x.extend(WINSPD[i:i+l])
    y = traffic[i+l-1]
    X.append(x)
    Y.append(y)
 
print ("###############")
print (np.corrcoef(CO, BC))
print (np.corrcoef(CO, NO2))
print (np.corrcoef(CO, NOX))
print (np.corrcoef(CO, PM25HR))
print (np.corrcoef(CO, TEMP))
print ("###############")
print (np.corrcoef(BC, NO2))
print (np.corrcoef(BC, NOX))
print (np.corrcoef(BC, PM25HR))
print (np.corrcoef(BC, TEMP))
print ("###############")
print (np.corrcoef(NO2, NOX))
print (np.corrcoef(NO2, PM25HR))
print (np.corrcoef(NO2, TEMP))
print ("###############")
print (np.corrcoef(NOX, PM25HR))
print (np.corrcoef(NOX, TEMP))
print ("###############")
print (np.corrcoef(PM25HR, TEMP))

print ("###############")
print (np.corrcoef(traffic, CO))
print (np.corrcoef(traffic, BC))
print (np.corrcoef(traffic, NO2))
print (np.corrcoef(traffic, NOX))
print (np.corrcoef(traffic, PM25HR))
print (np.corrcoef(traffic, TEMP))
print (np.corrcoef(traffic, WINSPD))


with open("input_data.csv", mode="w") as f:
    writer = csv.writer(f)
    for row in X:
        writer.writerow(row)
        
with open("output_data.csv", mode="w") as f:
    writer = csv.writer(f)
    writer.writerow(Y)
            

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
  











