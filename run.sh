#!/bin/bash
source ~/tensorflow/bin/activate
python MLP.py CO
python MLP.py BC
python MLP.py NO2
python MLP.py NOX
python MLP.py PM25HR

python LSTM.py CO
python LSTM.py BC
python LSTM.py NO2
python LSTM.py NOX
python LSTM.py PM25HR

python CNN.py CO
python CNN.py BC
python CNN.py NO2
python CNN.py NOX
python CNN.py PM25HR

python CNN_LSTM.py CO
python CNN_LSTM.py BC
python CNN_LSTM.py NO2
python CNN_LSTM.py NOX
python CNN_LSTM.py PM25HR

python ResNet.py CO
python ResNet.py BC
python ResNet.py NO2
python ResNet.py NOX
python ResNet.py PM25HR




