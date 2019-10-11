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

def loadallDF(list_cities, mdnRnA):
    import pandas as pd
    output = {}
    for city in list_cities:
        output[city + '_arima'] = pd.read_csv('../Data/Daily/{}/{}_arima_weekly.csv'.format(city, city),
                                         usecols=['fecha', 'tmed', 'presmed', 'velmedia'])
        output[city + '_arima']['mdnRnA'] = mdnRnA
        output[city + '_arima']['dates'] = pd.to_datetime(output[city + '_arima']['fecha'])
    return output

def corr(DF_list, arr_str, mdnRnA):
    for i in range(len(DF_list)):
        print("#################################################")
        print("Correlation(Temperature, Rn) in {}: ".format(arr_str[i]), pearsonr(mdnRnA, DF_list[i]['tmed'])[0])
        print("Correlation(Preassure, Rn) in {}: ".format(arr_str[i]), pearsonr(mdnRnA, DF_list[i]['presmed'])[0])
        print("Correlation(Wind velocity, Rn) in {}: ".format(arr_str[i]), pearsonr(mdnRnA, DF_list[i]['velmedia'])[0])
        print("#################################################")


def scaleallDF(DF_arr, arr_str):
    field = ['tmed', 'velmedia', 'presmed', 'mdnRnA']
    output = {}
    for i in range(len(DF_arr)):
        scaled = MinMaxScaler().fit(DF_arr[i][field].values).transform(DF_arr[i][field].values)
        output[arr_str[i]] = pd.DataFrame(scaled, columns=field)
        output[arr_str[i]]['dates'] = DF_arr[i]['dates']
    return output

def data_toCNN_format(DF_list, arr_str, fields, sample_size):
    output = {}
    for i in range(len(DF_list)):
        weekly_to3d = DF_list[i][fields].values
        output[arr_str[i]] = np.array([weekly_to3d[start:start+sample_size] for start in range(0, weekly_to3d.shape[0]-sample_size)])
    return output

def train_test_split(X, test_size):
    X_train, X_test = X[:-test_size], X[-test_size:]
    return X_train, X_test

def NN(neurons, nep, X_train, Y_train, X_test, Y_test, sample_size, v=0, btch_size=10):
    model = Sequential()
    model.add(Conv1D(filters=neurons[0], kernel_size=3, activation='relu', input_shape=X_train.shape[1:]))
    model.add(Flatten())
    model.add(Dense(neurons[0], activation='relu'))
    model.add(Dense(neurons[1], activation='relu'))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam", metrics=["acc"])
    history = model.fit(X_train, Y_train, epochs=nep, batch_size=btch_size, verbose=v, validation_data=(X_test, Y_test))
    pred = model.predict(X_test)
    acc_train = np.average(history.history["acc"])
    acc_test = np.average(history.history["val_acc"])
    #print("Train Accuracy: ", acc_train, "\nTest Accuracy:  ", acc_test)
    return history, pred, acc_train, acc_test

def show_errors(neurons, Xtrainlist, Y_train, Xtest_list, Y_test, arr_str, iterations, sample_size):
    for i in range(len(Xtrainlist)):
        print('\n\n#########\n', arr_str[i], '\n########\n\n')
        ECM = []
        EAM = []
        for it in range(iterations):
            #print('Iteration ', it)
            history, pred, acc_train, acc_test = NN(neurons, 30, Xtrainlist[i], Y_train,
                                                    Xtest_list[i], Y_test, sample_size)
            ECM.append(mean_squared_error(Y_test, pred))
            EAM.append(mean_absolute_error(Y_test, pred))
        print(':ECM: ', ECM)
        print(':EAM: ', EAM)
        print(':ECM avg: ', np.mean(ECM))
        print(':EAM avg: ', np.mean(EAM))

def Join_DF_RnT(*args):
    output = copy(args[0])
    for i in range(1, len(args)):
        output['tmed'+str(i)] = args[i]['tmed']
    return output