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
from copy import copy
from utils.NNutils import *

# Load:
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

# Scale:
weekly_scaled = scaleallDF(DF_list, arr_str)
BCN_scaled = weekly_scaled['BCN']
PMP_scaled = weekly_scaled['PMP']
HSC_scaled = weekly_scaled['HSC']
ZGZ_scaled = weekly_scaled['ZGZ']
DFscaled_list = [BCN_scaled, PMP_scaled, ZGZ_scaled, HSC_scaled]

# Useful parameters:
sample_size = 4
neuron = [64, 32]
test_size=int(0.3 * len(mdnRnA))
Y = mdnRnA[sample_size:]
Ytrain, Ytest = Y[:-test_size], Y[-test_size:]

###########################################################
#                  Couples of cities                      #
###########################################################
BCN_PMP_scaled = Join_DF_RnT(BCN_scaled, PMP_scaled)
BCN_HSC_scaled = Join_DF_RnT(BCN_scaled, HSC_scaled)
BCN_ZGZ_scaled = Join_DF_RnT(BCN_scaled, ZGZ_scaled)
PMP_HSC_scaled = Join_DF_RnT(PMP_scaled, HSC_scaled)
PMP_ZGZ_scaled = Join_DF_RnT(PMP_scaled, ZGZ_scaled)
HSC_ZGZ_scaled = Join_DF_RnT(HSC_scaled, ZGZ_scaled)
DFscaled_list_couples = [BCN_PMP_scaled, BCN_HSC_scaled, BCN_ZGZ_scaled, PMP_HSC_scaled, PMP_ZGZ_scaled, HSC_ZGZ_scaled]
arr_str_couples = ['BCN_PMP', 'BCN_HSC', 'BCN_ZGZ', 'PMP_HSC', 'PMP_ZGZ', 'HSC_ZGZ'] 
#Rn + Temperature
Xt = data_toCNN_format(DFscaled_list_couples, arr_str_couples, ['tmed', 'tmed1','mdnRnA'], sample_size)
Xt_BCN_PMP = Xt['BCN_PMP']
Xt_BCN_HSC = Xt['BCN_HSC']
Xt_BCN_ZGZ = Xt['BCN_ZGZ']
Xt_PMP_HSC = Xt['PMP_HSC']
Xt_PMP_ZGZ = Xt['PMP_ZGZ']
Xt_HSC_ZGZ = Xt['HSC_ZGZ']
# Train test split:
Xtrain_BCN_PMP, Xtest_BCN_PMP = train_test_split(Xt_BCN_PMP, test_size)
Xtrain_BCN_HSC, Xtest_BCN_HSC = train_test_split(Xt_BCN_HSC, test_size)
Xtrain_BCN_ZGZ, Xtest_BCN_ZGZ = train_test_split(Xt_BCN_ZGZ, test_size)
Xtrain_PMP_HSC, Xtest_PMP_HSC = train_test_split(Xt_PMP_HSC, test_size)
Xtrain_PMP_ZGZ, Xtest_PMP_ZGZ = train_test_split(Xt_PMP_ZGZ, test_size)
Xtrain_HSC_ZGZ, Xtest_HSC_ZGZ = train_test_split(Xt_HSC_ZGZ, test_size)
Xtrain_list = [Xtrain_BCN_PMP, Xtrain_BCN_HSC, Xtrain_BCN_ZGZ, Xtrain_PMP_HSC, Xtrain_PMP_ZGZ, Xtrain_HSC_ZGZ]
Xtest_list = [Xtest_BCN_PMP, Xtest_BCN_HSC, Xtest_BCN_ZGZ, Xtest_PMP_HSC, Xtest_PMP_ZGZ, Xtest_HSC_ZGZ]
# Predict and show errors:
show_errors([64, 32], Xtrain_list, Ytrain, Xtest_list, Ytest, arr_str_couples, iterations=25, sample_size=sample_size)