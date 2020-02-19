__author__ = '@Tssp'

''' RNN Bidireccional en Keras '''

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D
import pandas as pd  
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats.stats import pearsonr
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates
from utils.NNutils import *

# Load:
## 222Rn
mdnRnA = np.loadtxt('../../mdnRnA.txt', delimiter=',')
startday = pd.datetime(2013, 7, 1)
dates = pd.date_range(startday, periods=len(mdnRnA), freq='W')
## Weather:
list_cities = ['BCN', 'NVR', 'HSC', 'ZGZ']
weekly = loadallDF(list_cities, mdnRnA)
BCN_arima = weekly['BCN_arima']
NVR_arima = weekly['NVR_arima']
HSC_arima = weekly['HSC_arima']
ZGZ_arima = weekly['ZGZ_arima']
DF_list = [BCN_arima, NVR_arima, ZGZ_arima, HSC_arima]
arr_str = ['BCN', 'PMP', 'ZGZ', 'HSC']

# Correlations:
corr(DF_list, arr_str, mdnRnA)

# Scale DataFrame
weekly_scaled = scaleallDF(DF_list, arr_str)
BCN_scaled = weekly_scaled['BCN']
PMP_scaled = weekly_scaled['PMP']
HSC_scaled = weekly_scaled['HSC']
ZGZ_scaled = weekly_scaled['ZGZ']
DFscaled_list = [BCN_scaled, PMP_scaled, ZGZ_scaled, HSC_scaled]

# Only Rn
sample_size = 4
test_size=int(0.3 * len(mdnRnA))
X = np.atleast_3d(np.array([mdnRnA[start:start + sample_size] for start in range(0, mdnRnA.shape[0]-sample_size)]))
Y = mdnRnA[sample_size:]
Xtrain, Xtest = X[:-test_size], X[-test_size:]
Ytrain, Ytest = Y[:-test_size], Y[-test_size:]
print("X_train.shape = ", Xtrain.shape, "\nY_train.shape = ", Ytrain.shape)
### Predict
ECM = []
EAM = []
print('\n\n#########\n Only Radon \n########\n\n')
for it in range(25):
    #print('Iteration ', it)
    history, pred, acc_train, acc_test = NN([64, 32], nep=35, X_train=Xtrain, Y_train=Ytrain,
                                            X_test=Xtest, Y_test=Ytest, sample_size=sample_size)
    ECM.append(mean_squared_error(Ytest, pred))
    EAM.append(mean_absolute_error(Ytest, pred))
print(':ECM: ', ECM)
print(':EAM: ', EAM)
print(':ECM avg: ', np.mean(ECM))
print(':EAM avg: ', np.mean(EAM))

#Rn + Temperature
print('\n\n#########\n Radon + Temperature \n########\n\n')
Xt = data_toCNN_format(DFscaled_list, arr_str, ['tmed', 'mdnRnA'], sample_size)
Xt_BCN = Xt['BCN']
Xt_PMP = Xt['PMP']
Xt_HSC = Xt['HSC']
Xt_ZGZ = Xt['ZGZ']
Xtrain_BCN, Xtest_BCN = train_test_split(Xt_BCN, test_size)
Xtrain_PMP, Xtest_PMP = train_test_split(Xt_PMP, test_size)
Xtrain_HSC, Xtest_HSC = train_test_split(Xt_HSC, test_size)
Xtrain_ZGZ, Xtest_ZGZ = train_test_split(Xt_ZGZ, test_size)
## Predict
Xtrainlist = [Xtrain_BCN, Xtrain_PMP, Xtrain_HSC, Xtrain_ZGZ]
Xtestlist = [Xtest_BCN, Xtest_PMP, Xtest_ZGZ, Xtest_HSC]
show_errors([64, 32], Xtrainlist, Ytrain, Xtestlist, Ytest, arr_str, iterations=25, sample_size=sample_size)

#Rn + Preassure
print('\n\n#########\n Radon + Preassure \n########\n\n')
Xp = data_toCNN_format(DFscaled_list, arr_str, ['presmed', 'mdnRnA'], sample_size)
Xp_BCN = Xp['BCN']
Xp_PMP = Xp['PMP']
Xp_HSC = Xp['HSC']
Xp_ZGZ = Xp['ZGZ']
Xtrain_BCN, Xtest_BCN = train_test_split(Xp_BCN, test_size)
Xtrain_PMP, Xtest_PMP = train_test_split(Xp_PMP, test_size)
Xtrain_HSC, Xtest_HSC = train_test_split(Xp_HSC, test_size)
Xtrain_ZGZ, Xtest_ZGZ = train_test_split(Xp_ZGZ, test_size)
## Predict
Xtrainlist = [Xtrain_BCN, Xtrain_PMP, Xtrain_HSC, Xtrain_ZGZ]
Xtestlist = [Xtest_BCN, Xtest_PMP, Xtest_ZGZ, Xtest_HSC]
show_errors([64, 32], Xtrainlist, Ytrain, Xtestlist, Ytest, arr_str, iterations=25, sample_size=sample_size)

# Rn + wind velocity
print('\n\n#########\n Radon + Wind Velocity \n########\n\n')
Xv = data_toCNN_format(DFscaled_list, arr_str, ['velmedia', 'mdnRnA'], sample_size)
Xv_BCN = Xv['BCN']
Xv_PMP = Xv['PMP']
Xv_HSC = Xv['HSC']
Xv_ZGZ = Xv['ZGZ']
Xtrain_BCN, Xtest_BCN = train_test_split(Xv_BCN, test_size)
Xtrain_PMP, Xtest_PMP = train_test_split(Xv_PMP, test_size)
Xtrain_HSC, Xtest_HSC = train_test_split(Xv_HSC, test_size)
Xtrain_ZGZ, Xtest_ZGZ = train_test_split(Xv_ZGZ, test_size)
## Predict
Xtrainlist = [Xtrain_BCN, Xtrain_PMP, Xtrain_HSC, Xtrain_ZGZ]
Xtestlist = [Xtest_BCN, Xtest_PMP, Xtest_ZGZ, Xtest_HSC]
show_errors([64, 32], Xtrainlist, Ytrain, Xtestlist, Ytest, arr_str, iterations=25, sample_size=sample_size)






