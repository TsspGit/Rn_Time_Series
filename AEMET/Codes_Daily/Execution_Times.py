''' RNN Bidireccional en Keras '''
__author_ = '@Tssp'
import sys as sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D, AveragePooling1D
import pandas as pd  
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats.stats import pearsonr
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates
from utils.NNutils import *
from time import time
plt.rcParams['xtick.labelsize']=13
plt.rcParams['ytick.labelsize']=13
plt.rcParams['axes.labelsize']=16
plt.rcParams['axes.titlesize']=16

# ¡Comment and discomment the Neural Network as you want! 

###########################################################
#                  Only Rn                                #
###########################################################
mdnRnA = np.loadtxt('../../mdnRnA.txt', delimiter=',')
forward = 8
newValuesReal = np.loadtxt('../../mdnRnA_validation.txt', delimiter=',')[:forward]
nlags= len(newValuesReal)-1
startday = pd.datetime(2013, 7, 1)
dates = pd.date_range(startday, periods=len(mdnRnA), freq='W')
# Weather
NVR = pd.read_csv('../Data/Daily/NVR/NVR_arima_weekly.csv', usecols=['fecha', 'tmed', 'presmed', 'velmedia'])
NVR['fecha'] = pd.to_datetime(NVR['fecha'])
DF = pd.DataFrame({'dates': dates,'mdnRnA': mdnRnA, 'tmed': NVR['tmed'].values})
sample_size = 52
# dataframe (empty) creation
pdata = pd.DataFrame()
pdatamdnRnA = pd.DataFrame()
names=list()
nlags= len(newValuesReal)-1
for i in range(nlags, -1, -1):
    # Add the new lagged column at the end of the dataframe. In inverse order.
    pdata = pd.concat([pdata, DF[['tmed', 'mdnRnA']].shift(i).reset_index(drop=True)], axis=1)
    pdatamdnRnA = pd.concat([pdatamdnRnA, DF['mdnRnA'].shift(i).reset_index(drop=True)], axis=1)
    names += str(i) # store this index for naming the columns of dataframe
pdatalags = np.asarray(pdata[nlags:])
pdatamdnRnAlags = np.asarray(pdatamdnRnA[nlags:])   
X = np.atleast_3d(np.array([pdatamdnRnAlags[start:start + sample_size]
    for start in range(0, pdatamdnRnAlags.shape[0]-sample_size)]))
Y = pdatamdnRnAlags[sample_size:]
test_size = int(0.3*len(mdnRnA))
X_train,X_test = X[:-test_size], X[-test_size:]
Y_train,Y_test = Y[:-test_size], Y[-test_size:]

# CNN:
neuronCNN = [64, 128]
timeCNN = []
for it in range(10):
    start = time()
    history, pred, acc_train, acc_test, model = NN_v2(neuronCNN, 40, X_train, Y_train, X_test, Y_test, sample_size)
    timeCNN.append(time() - start)
# ANN:
    
neuronANN = [256, 128]
timeANN = []
for it in range(10):
    start = time()
    history, pred, acc_train, acc_test, model = ANN_v2(neuronANN, 40, X_train, Y_train, X_test, Y_test, sample_size)
    timeANN.append(time() - start)
# RNN:
neuron = [52, 64]
timeRNN = []
for it in range(10):
    start = time()
    history, pred, acc_train, acc_test, model = RNN_v2(neuron, 7, X_train, Y_train, X_test, Y_test, sample_size)
    timeRNN.append(time() - start)
    
###########################################################
#                   Independent Cities                    #
###########################################################
list_cities = ['BCN', 'NVR', 'HSC', 'ZGZ']
weekly = loadallDF(list_cities, mdnRnA)
BCN_arima = weekly['BCN_arima']
NVR_arima = weekly['NVR_arima']
HSC_arima = weekly['HSC_arima']
ZGZ_arima = weekly['ZGZ_arima']
DF_list = [BCN_arima, NVR_arima, ZGZ_arima, HSC_arima]
arr_str = ['BCN', 'PMP', 'ZGZ', 'HSC']
#Rn + Temperature
DFscaled_list, pdatamdnRnAlags = build_pdata(DF_list, DF, nlags, ['tmed', 'mdnRnA'])
Xt = data_toCNN_format_v2(DFscaled_list, arr_str, ['tmed', 'mdnRnA'], sample_size, pdatalags)
Xt_BCN = Xt['BCN']
Xt_PMP = Xt['PMP']
Xt_HSC = Xt['HSC']
Xt_ZGZ = Xt['ZGZ']
Xtrain_BCN, Xtest_BCN = train_test_split(Xt_BCN, test_size)
Xtrain_PMP, Xtest_PMP = train_test_split(Xt_PMP, test_size)
Xtrain_HSC, Xtest_HSC = train_test_split(Xt_HSC, test_size)
Xtrain_ZGZ, Xtest_ZGZ = train_test_split(Xt_ZGZ, test_size)
Ytrain, Ytest = Y[:-test_size], Y[-test_size:]
Xtrainlist = [Xtrain_HSC]
Xtestlist = [Xtest_HSC]
arr_str = ['HSC']

# CNN:
neuronCNN = [128, 256]
timeCNNT = []
for it in range(10):
    start = time()
    history, pred, acc_train, acc_test, model = NN_v2(neuronCNN, 80, Xtrainlist[0], Ytrain, Xtestlist[0], Ytest, sample_size)
    timeCNNT.append(time() - start)
    
# ANN:
timeANNT = []
for it in range(10):
    start = time()
    history, pred, acc_train, acc_test, model = ANN_v2(neuronANN, 60, Xtrainlist[0], Ytrain, Xtestlist[0], Ytest, sample_size)
    timeANNT.append(time() - start)
    
# RNN:
neuron = [52, 104]
timeRNNT = []
for it in range(10):
    start = time()
    history, pred, acc_train, acc_test, model = RNN_v2(neuron, 9, Xtrainlist[0], Ytrain, Xtestlist[0], Ytest, sample_size)
    timeRNNT.append(time() - start)

times = pd.DataFrame({'tCNN' : timeCNN,
                     'tANN' : timeANN,
                     'tRNN' : timeRNN,
                     'tCNNT' : timeCNNT,
                     'tANNT' : timeANNT,
                     'tRNNT' : timeRNNT,
                     })
times.to_csv('Times.csv', index=False)
