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
neuron = [64, 32]
test_size=int(0.3 * len(mdnRnA))
X = np.atleast_3d(np.array([mdnRnA[start:start + sample_size] for start in range(0, mdnRnA.shape[0]-sample_size)]))
Y = mdnRnA[sample_size:]
Xtrain, Xtest = X[:-test_size], X[-test_size:]
Ytrain, Ytest = Y[:-test_size], Y[-test_size:]
print("X_train.shape = ", Xtrain.shape, "\nY_train.shape = ", Ytrain.shape)

## Predict
history, pred, acc_train, acc_test = NN(neuron, nep=30, X_train=Xtrain, Y_train=Ytrain, X_test=Xtest, Y_test=Ytest, sample_size=sample_size)

## Errors
testScoreECM = mean_squared_error(Ytest, pred)
print('ECM: %.4f' % (testScoreECM))
testScoreEAM = mean_absolute_error(Ytest, pred)
print('EAM: %.4f' % (testScoreEAM))

## Plot
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss', fontsize=14)
plt.xlabel('epoch', fontsize=14)
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
#plt.savefig('./CNN_Loss_Rn_{}_{}.png'.format(neuron[0], neuron[1]))
np.savetxt('CNN_Loss_Rn_{}_{}_v2.txt'.format(neuron[0], neuron[1]), (history.history['loss'], history.history['val_loss']), delimiter=',')

## Plot2
startdaypred = pd.datetime(2013, 7, 1) + 7*pd.Timedelta( len(mdnRnA)-len(pred), unit='D')
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,4), constrained_layout=True)
xaxis = ax.get_xaxis()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax.plot(pd.date_range(startday, periods=len(mdnRnA), freq='W'), mdnRnA, 'k', alpha=0.7) 
ax.plot(pd.date_range(startdaypred, periods=len(pred), freq='W'), pred, linewidth=2, linestyle='-',color='crimson')
plt.xlabel('Dates', fontsize=16)
plt.ylabel(r'$^{222}Rn\ (Bq/m^3)$', fontsize=16)
ax.legend(['Data', 'CNN'], loc='upper left')
plt.ylim([30, 140])

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
axins = zoomed_inset_axes(ax, 1.7, loc='lower left', bbox_to_anchor=(2361,700))
axins.xaxis.set_major_locator(mdates.YearLocator())
axins.xaxis.set_minor_locator(mdates.MonthLocator())
axins.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axins.plot(pd.date_range(startday, periods=len(mdnRnA), freq='W'), mdnRnA, 'k', alpha=0.7) 
axins.plot(pd.date_range(startdaypred, periods=len(pred), freq='W'), pred, linewidth=2, linestyle='-',color='crimson')
axins.set_xlim('2017-10-05', '2019-07-21')
axins.set_ylim(50, 110)
plt.xticks(visible=True)
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.suptitle('Weekly Fitting at LSC - Hall A')
#fig.savefig('./CNN_Rn_{}_{}.png'.format(neuron[0], neuron[1]), bbox_inches='tight', dpi=300)
np.savetxt('CNN_Rn_{}_{}_v2.txt'.format(neuron[0], neuron[1]), pred, delimiter=',', fmt='%s')


#Rn + Temperature
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
history, pred, acc_train, acc_test = NN(neuron, nep=30, X_train=Xtrain_PMP, Y_train=Ytrain, X_test=Xtest_PMP, Y_test=Ytest, sample_size=sample_size, save=True)

## Errors
testScoreECM = mean_squared_error(Ytest, pred)
print('ECM: %.4f' % (testScoreECM))
testScoreEAM = mean_absolute_error(Ytest, pred)
print('EAM: %.4f' % (testScoreEAM))

## Plot
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss', fontsize=14)
plt.xlabel('epoch', fontsize=14)
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
#plt.savefig('./CNN_Loss_RnT_PMP_{}_{}.png'.format(neuron[0], neuron[1]))
np.savetxt('CNN_Loss_RnT_PMP_{}_{}_v2.txt'.format(neuron[0], neuron[1]), (history.history['loss'], history.history['val_loss']), delimiter=',')

## Plot2
startdaypred = pd.datetime(2013, 7, 1) + 7*pd.Timedelta( len(mdnRnA)-len(pred), unit='D')
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,4))
xaxis = ax.get_xaxis()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax.plot(pd.date_range(startday, periods=len(mdnRnA), freq='W'), mdnRnA, 'k', alpha=0.7) 
ax.plot(pd.date_range(startdaypred, periods=len(pred), freq='W'), pred, linewidth=2, linestyle='-',color='crimson')
plt.xlabel('Dates', fontsize=16)
plt.ylabel(r'$^{222}Rn\ (Bq/m^3)$', fontsize=16)
ax.legend(['Data', 'CNN(Rn + T)'], loc='upper left')
plt.ylim([30, 140])

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
axins = zoomed_inset_axes(ax, 1.7, loc='lower left', bbox_to_anchor=(2361,700))
axins.xaxis.set_major_locator(mdates.YearLocator())
axins.xaxis.set_minor_locator(mdates.MonthLocator())
axins.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axins.plot(pd.date_range(startday, periods=len(mdnRnA), freq='W'), mdnRnA, 'k', alpha=0.7) 
axins.plot(pd.date_range(startdaypred, periods=len(pred), freq='W'), pred, linewidth=2, linestyle='-',color='crimson')
axins.set_xlim('2017-10-05', '2019-07-21')
axins.set_ylim(50, 110)
plt.xticks(visible=True)
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.suptitle('Weekly Fitting at LSC - Hall A')
#fig.savefig('./CNN_RnT_PMP_{}_{}.png'.format(neuron[0], neuron[1]), bbox_inches='tight', dpi=300)
np.savetxt('CNN_RnT_PMP_{}_{}_v2.txt'.format(neuron[0], neuron[1]), pred, delimiter=',', fmt='%s')




