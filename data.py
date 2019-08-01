import numpy as np
import csv


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
                data.insert( index + 1 + j, (d1, h1 + 1 + j, 0))
            index = index + ml
    
        if ((d1 != d2) and ( h1 != 23)):
            ml = 23 - h1 
            for j in range(0, ml):
                #print("check")
                data.insert( index + 1 + j, (d1, h1 + 1 + j, 0))
            index = index + ml
            
        if ((d1 != d2) and ( h2 != 0)):
            #print("###############")
            #print d1
            #print d2
            ml = h2 
            for j in range(0, ml):
                data.insert( index + 1 + j, (d2, j, 0))
            index = index + ml
        L = len(data)
    
    if (L!= 31*24):
        ml = 31*24 - L
        for j in range(0, ml):
            data.insert(L + j, (data[L-1][0], data[L-1][1] +1 + j, 0))
    
        
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

CO_h =  [i[2] for i in CO_list]
BC_h =  [i[2] for i in BC_list]
NO2_h = [i[2] for i in NO2_list]
NOX_h = [i[2] for i in NOX_list]
PM25HR_h = [i[2] for i in PM25HR_list]
TEMP_h =   [i[2] for i in TEMP_list]

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





