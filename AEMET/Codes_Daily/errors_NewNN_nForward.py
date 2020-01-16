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
plt.rcParams['xtick.labelsize']=13
plt.rcParams['ytick.labelsize']=13
plt.rcParams['axes.labelsize']=16
plt.rcParams['axes.titlesize']=16

# ¡Comment and discomment the Neural Network as you want! 

###########################################################
#                  Only Rn                                #
###########################################################
mdnRnA = np.loadtxt('../../mdnRnA.txt', delimiter=',')
forward = int(sys.argv[1])
newValuesReal = np.loadtxt('../../mdnRnA_validation.txt', delimiter=',')[:forward]
nlags= len(newValuesReal)-1
startday = pd.datetime(2013, 7, 1)
dates = pd.date_range(startday, periods=len(mdnRnA), freq='W')
# Weather
NVR = pd.read_csv('../Data/Daily/NVR/NVR_arima_weekly.csv', usecols=['fecha', 'tmed', 'presmed', 'velmedia'])
NVR['fecha'] = pd.to_datetime(NVR['fecha'])
DF = pd.DataFrame({'dates': dates,'mdnRnA': mdnRnA, 'tmed': NVR['tmed'].values})
def scaler(DF, field):
    scaled = MinMaxScaler().fit(DF[[field[0], field[1]]].values).transform(DF[[field[0], field[1]]].values)
    output = pd.DataFrame(scaled, columns=['tmed', 'mdnRnA'])
    output['dates'] = DF['dates']
    return output, scaler
DF_scaled, scaled = scaler(DF, ['tmed', 'mdnRnA'])
sample_size = 52
# dataframe (empty) creation
pdata = pd.DataFrame()
pdatamdnRnA = pd.DataFrame()
names=list()
nlags= len(newValuesReal)-1
for i in range(nlags, -1, -1):
    # Add the new lagged column at the end of the dataframe. In inverse order.
    pdata = pd.concat([pdata, DF_scaled[['tmed', 'mdnRnA']].shift(i).reset_index(drop=True)], axis=1)
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
neuronCNN = [128, 64]
neuronANN = [64, 32]
ECM = []
EAM = []
for it in range(25):
    #history, pred, acc_train, acc_test, model = NN_v2(neuronCNN, 80, X_train, Y_train, X_test, Y_test, sample_size)
    history, pred, acc_train, acc_test, model = ANN_v2(neuronANN, 40, X_train, Y_train, X_test, Y_test, sample_size)
    predmaxs, predmins, predavgs = extract_maxs_mins_avgs(pred)
    Y_test_error = mdnRnA[-len(predavgs):]
    ECM.append(mean_squared_error(Y_test_error, predavgs))
    EAM.append(mean_absolute_error(Y_test_error, predavgs))
print('ECMRn = ', ECM)
print('EAMRn = ', EAM)

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
# Scale DataFrame
weekly_scaled = scaleallDF(DF_list, arr_str)
BCN_scaled = weekly_scaled['BCN']
PMP_scaled = weekly_scaled['PMP']
HSC_scaled = weekly_scaled['HSC']
ZGZ_scaled = weekly_scaled['ZGZ']
DFscaled_list = [BCN_scaled, PMP_scaled, ZGZ_scaled, HSC_scaled]
#Rn + Temperature
print('\n\n#########\n Radon + Temperature \n########\n\n')
DFscaled_list, pdatamdnRnAlags = build_pdata(DFscaled_list, DF, nlags, ['tmed', 'mdnRnA'])
Xt = data_toCNN_format_v2(DFscaled_list, arr_str, ['tmed', 'mdnRnA'], sample_size, pdatalags)
Xt_BCN = Xt['BCN']
Xt_PMP = Xt['PMP']
Xt_HSC = Xt['HSC']
Xt_ZGZ = Xt['ZGZ']
Y = pdatamdnRnAlags[sample_size:]
Xtrain_BCN, Xtest_BCN = train_test_split(Xt_BCN, test_size)
Xtrain_PMP, Xtest_PMP = train_test_split(Xt_PMP, test_size)
Xtrain_HSC, Xtest_HSC = train_test_split(Xt_HSC, test_size)
Xtrain_ZGZ, Xtest_ZGZ = train_test_split(Xt_ZGZ, test_size)
Ytrain, Ytest = Y[:-test_size], Y[-test_size:]
Xtrainlist = [Xtrain_BCN, Xtrain_PMP, Xtrain_HSC, Xtrain_ZGZ]
Xtestlist = [Xtest_BCN, Xtest_PMP, Xtest_ZGZ, Xtest_HSC]
neuronCNN = [256, 128]
#show_errors_v2(neuronCNN, Xtrainlist, Ytrain, Xtestlist, Ytest, arr_str, iterations=25, sample_size=sample_size, DF_mdnRnA=DF)
show_errors_ANNv2(neuronCNN, Xtrainlist, Ytrain, Xtestlist, Ytest, arr_str, iterations=25, sample_size=sample_size, DF_mdnRnA=DF)

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
DFscaled_list_couples, pdatamdnRnAlags = build_pdata(DFscaled_list_couples, DF, nlags, ['tmed', 'tmed1','mdnRnA'])
Xt = data_toCNN_format_v2(DFscaled_list_couples, arr_str_couples, ['tmed', 'tmed1', 'mdnRnA'], sample_size, pdatalags)
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
Xtrainlist = [Xtrain_BCN_PMP, Xtrain_BCN_HSC, Xtrain_BCN_ZGZ, Xtrain_PMP_HSC, Xtrain_PMP_ZGZ, Xtrain_HSC_ZGZ]
Xtestlist = [Xtest_BCN_PMP, Xtest_BCN_HSC, Xtest_BCN_ZGZ, Xtest_PMP_HSC, Xtest_PMP_ZGZ, Xtest_HSC_ZGZ]
# show_errors_v2(neuronCNN, Xtrainlist, Ytrain, Xtestlist, Ytest, arr_str_couples, iterations=25, sample_size=sample_size, DF_mdnRnA=DF)
show_errors_ANNv2(neuronCNN, Xtrainlist, Ytrain, Xtestlist, Ytest, arr_str_couples, iterations=25, sample_size=sample_size, DF_mdnRnA=DF)

###########################################################
#                  Trios of cities                        #
###########################################################
BCN_PMP_HSC_scaled = Join_DF_RnT(BCN_scaled, PMP_scaled, HSC_scaled)
BCN_PMP_ZGZ_scaled = Join_DF_RnT(BCN_scaled, PMP_scaled, ZGZ_scaled)
PMP_HSC_ZGZ_scaled = Join_DF_RnT(PMP_scaled, HSC_scaled, ZGZ_scaled)
DFscaled_list_trios = [BCN_PMP_HSC_scaled, BCN_PMP_ZGZ_scaled, PMP_HSC_ZGZ_scaled]
arr_str_trios = ['BCN_PMP_HSC', 'BCN_PMP_ZGZ', 'PMP_HSC_ZGZ'] 
DFscaled_list_trios, pdatamdnRnAlags = build_pdata(DFscaled_list_trios, DF, nlags, ['tmed', 'tmed1', 'tmed2', 'mdnRnA'])
Xt = data_toCNN_format_v2(DFscaled_list_trios, arr_str_trios, ['tmed', 'tmed1','tmed2', 'mdnRnA'], sample_size, pdatalags)
Xt_BCN_PMP_HSC = Xt['BCN_PMP_HSC']
Xt_BCN_PMP_ZGZ = Xt['BCN_PMP_ZGZ']
Xt_PMP_HSC_ZGZ = Xt['PMP_HSC_ZGZ']
Xtrain_BCN_PMP_HSC, Xtest_BCN_PMP_HSC = train_test_split(Xt_BCN_PMP_HSC, test_size)
Xtrain_BCN_PMP_ZGZ, Xtest_BCN_PMP_ZGZ = train_test_split(Xt_BCN_PMP_ZGZ, test_size)
Xtrain_PMP_HSC_ZGZ, Xtest_PMP_HSC_ZGZ = train_test_split(Xt_PMP_HSC_ZGZ, test_size)
Xtrain_list = [Xtrain_BCN_PMP_HSC, Xtrain_BCN_PMP_ZGZ, Xtrain_PMP_HSC_ZGZ]
Xtest_list = [Xtest_BCN_PMP_HSC, Xtest_BCN_PMP_ZGZ, Xtest_PMP_HSC_ZGZ]
# show_errors_v2(neuronCNN, Xtrainlist, Ytrain, Xtestlist, Ytest, arr_str_trios, iterations=25, sample_size=sample_size, DF_mdnRnA=DF)
show_errors_ANNv2(neuronCNN, Xtrainlist, Ytrain, Xtestlist, Ytest, arr_str_trios, iterations=25, sample_size=sample_size, DF_mdnRnA=DF)