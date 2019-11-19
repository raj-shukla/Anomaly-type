import numpy as np
from sklearn.preprocessing import Imputer

import csv


nan = np.nan

def read_traffic(param_file):
    data = []
    csvFile = open (param_file, "r")
    csvReader = csv.reader(csvFile)
    next(csvReader)
    for row in csvReader:
        data.append(float(row[1]))
    
    return data
    


def read_data(param_file):
    data = []
    csvFile = open (param_file, "r")
    csvReader = csv.reader(csvFile)
    next(csvReader)
    for row in csvReader:
        if (len(row) < 9 or row[2] == ''):
            break 
        if (param_file == "Data/WINSPD_PICKDATA_2017-10-31.csv"):
            data.append((row[1], int(row[2]), row[3]))
        else:
            data.append((row[1], int(row[2]), float(row[3])))
    
    L = len(data)
    for i in range(0, L - 1):
        index = i
        d1= data[i][0]
        #print data[i]
        #print i
        #print (len(data))
        d2 = data[i+1][0]
        h1 = data[i][1]
        h2 = data[i+1][1]
  
        if ((d1 == d2) and ( (h2 - h1) != 1)):
            #print("################")
            #print(data[i])
            #print(data[i+1])
            ml = h2 - h1 - 1
            #print (ml)
            for j in range(0, ml):
                #print("check")
                data.insert( index + 1 + j, (d1, h1 + 1 + j, nan))
            index = index + ml
    
        if ((d1 != d2) and ( h1 != 23)):
            ml = 23 - h1 
            for j in range(0, ml):
                #print("check")
                data.insert( index + 1 + j, (d1, h1 + 1 + j, nan))
            index = index + ml
            
        if ((d1 != d2) and ( h2 != 0)):
            #print("###############")
            #print d1
            #print d2
            ml = h2 
            for j in range(0, ml):
                data.insert( index + 1 + j, (d2, j, nan))
            index = index + ml
        L = len(data)
    
    if (L!= 31*24):
        ml = 31*24 - L
        for j in range(0, ml):
            data.insert(L + j, (data[L-1][0], data[L-1][1] +1 + j, nan))
    
        
    return data
 

traffic_m = []
for i in range(0, 5):
    file_name = "Data/pems_output_" + str(i) + ".csv"
    data = read_traffic (file_name)
    traffic_m.extend(data)
    
       

CO_list = read_data("Data/CO_PICKDATA_2017-10-31.csv")
BC_list = read_data("Data/BC_PICKDATA_2017-10-31.csv")
NO2_list = read_data("Data/NO2_PICKDATA_2017-10-31.csv")
NOX_list = read_data("Data/NOX_PICKDATA_2017-10-31.csv")
PM25HR_list = read_data("Data/PM25HR_PICKDATA_2017-10-31.csv")
TEMP_list = read_data("Data/TEMP_PICKDATA_2017-10-31.csv")
WINSPD_list = read_data("Data/WINSPD_PICKDATA_2017-10-31.csv")


CO_h =  np.asarray([i[2] for i in CO_list]).reshape(31, 24)
BC_h =  np.asarray([i[2] for i in BC_list]).reshape(31, 24)
NO2_h = np.asarray([i[2] for i in NO2_list]).reshape(31, 24)
NOX_h = np.asarray([i[2] for i in NOX_list]).reshape(31, 24)
PM25HR_h = np.asarray([i[2] for i in PM25HR_list]).reshape(31, 24)
TEMP_h =   np.asarray([i[2] for i in TEMP_list]).reshape(31, 24)
WINSPD_tmp = [0 if i[2] == 'CALM' else i[2] for i in WINSPD_list]
WINSPD_h = np.asarray([float(i.split('/')[-1]) if type(i) == str else i  for i in WINSPD_tmp]).reshape(31, 24)

print (CO_h)

imp_mean = Imputer(missing_values=nan, strategy='mean')
CO_h = np.round(imp_mean.fit_transform(CO_h), 2).flatten()
BC_h = np.round(imp_mean.fit_transform(BC_h), 2).flatten()
NO2_h = np.round(imp_mean.fit_transform(NO2_h), 2).flatten()
NOX_h = np.round(imp_mean.fit_transform(NOX_h), 2).flatten()
PM25HR_h = np.round(imp_mean.fit_transform(PM25HR_h), 2).flatten()
TEMP_h = np.round(imp_mean.fit_transform(TEMP_h), 2).flatten()
WINSPD_h = np.round(imp_mean.fit_transform(WINSPD_h), 2).flatten()



print("################")
print(CO_h)
print("################")
print(BC_h)
print("################")
print(NO2_h)
print("################")
print(NOX_h)
print("################")
print(PM25HR_h)
print("################")
print(TEMP_h)
print("################")
print(WINSPD_h)


'''
data = []
csvFile = open ("Data/pems_output_2.csv", "r")
csvReader = csv.reader(csvFile)
next(csvReader)
for row in csvReader:
    data.append(int(row[0].split(':')[-1]))
    
print(len(data))

print(data[0:20])

for i in range(0, len(data) -1):
    b = data[i+1] - data[i]
    if(b!= 5 and b!=-55):
        print(i)

'''





