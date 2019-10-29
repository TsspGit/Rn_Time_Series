__author__ = '@Tssp'

''' RNN Bidireccional en Keras '''

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import model_from_json
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
        output[arr_str[i]] = np.array([weekly_to3d[start:start+sample_size] for start in range(0, weekly_to3d.shape[0]-sample_size+1)])
    return output

def data_toCNN_format_v2(DF_list, arr_str, fields, sample_size, pdatalags):
    output = {}
    for i in range(len(DF_list)):
        weekly_to3d = DF_list[i][fields].values
        output[arr_str[i]] = np.array([pdatalags[start:start+sample_size] for start in range(0, pdatalags.shape[0]-sample_size)])
    return output


def train_test_split(X, test_size):
    X_train, X_test = X[:-test_size], X[-test_size:]
    return X_train, X_test


def save_NN(model):
    model.save("modelaemet.h5")
    print("Save model to disk")
    
    
def NN(neurons, nep, X_train, Y_train, X_test, Y_test, sample_size, v=0, btch_size=10, save=False):
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
    if save:
        save_NN(model)
    return history, pred, acc_train, acc_test

def build_pdata(DF_list, DF, nlags, fields):
    names=list()
    out = []
    for city in DF_list:
        pdata = pd.DataFrame()
        pdatamdnRnA = pd.DataFrame()
        for i in range(nlags, -1, -1):
            # Add the new lagged column at the end of the dataframe. In inverse order.
            pdata = pd.concat([pdata, city[fields].shift(i).reset_index(drop=True)], axis=1)
            pdatamdnRnA = pd.concat([pdatamdnRnA, DF['mdnRnA'].shift(i).reset_index(drop=True)], axis=1)
            names += str(i) # store this index for naming the columns of dataframe
            pdatalags = np.asarray(pdata[nlags:])
            pdatamdnRnAlags = np.asarray(pdatamdnRnA[nlags:])
        out.append(pdata)
    return out, pdatamdnRnAlags

def NN_v2(neurons, nep, X_train, Y_train, X_test, Y_test, sample_size, v=0, btch_size=10, save=False):
    model = Sequential()
    model.add(Conv1D(filters=int(neurons[0]), kernel_size=3, activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv1D(filters=int(neurons[1]), kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(int(neurons[0]/2), activation='relu'))
    model.add(Dense(int(neurons[1]/2), activation='relu'))
    model.add(Dense(Y_train.shape[1], activation='linear'))
    model.compile(loss="mae", optimizer="adam", metrics=["acc"])
    history = model.fit(X_train, Y_train, epochs=nep, batch_size=btch_size, verbose=v, validation_data=(X_test, Y_test))
    pred = model.predict(X_test)
    acc_train = np.average(history.history["acc"])
    acc_test = np.average(history.history["val_acc"])
    if save:
        save_NN(model)
    return history, pred, acc_train, acc_test, model
def extract_maxs_mins_avgs(M):
    diagM = [M[::-1].diagonal(i) for i in range(-M.shape[0]+1, M.shape[1])]
    mins = []
    maxs = []
    avgs = []
    for row in range(len(diagM)):
        mins.append(np.min(diagM[row]))
        maxs.append(np.max(diagM[row]))
        avgs.append(np.mean(diagM[row]))
    return maxs, mins, avgs

def plot_forecast(data, startday, pred, startdaypred):
    import pandas as pd
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,4))
    plt.figure(1)
    xaxis = ax.get_xaxis()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.plot(pd.date_range(startday, periods=len(data), freq='W')[:-5], data[:-5], 'k') 
    ax.plot(pd.date_range(startdaypred, periods=len(pred), freq='W')[:-5], pred[:-5],
            linewidth=2, linestyle='-',color='crimson')

    plt.xlabel('Dates', fontsize=16)
    ax.legend(['Data', 'CNN (Rn + T)'], loc='upper left')
    plt.ylim([30, 140])

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
    axins = zoomed_inset_axes(ax, 1.7, loc='lower left', bbox_to_anchor=(643,140))
    axins.xaxis.set_major_locator(mdates.YearLocator())
    axins.xaxis.set_minor_locator(mdates.MonthLocator())
    axins.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axins.plot(pd.date_range(startday, periods=len(data), freq='W')[:-5], data[:-5],
               'k') 
    axins.plot(pd.date_range(startdaypred, periods=len(pred), freq='W')[:-5], pred[:-5], linewidth=2,
               linestyle='-',color='crimson')
    axins.set_xlim('2017-08-05', '2019-06-24')
    axins.set_ylim(50, 120)
    axins.set_yticks([60, 80, 100])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    
    plt.suptitle('Weekly Fitting at LSC - Hall A')
    plt.show()
    
def plot_validation(data_val, pred_val, startday_ahead):
    import matplotlib.pyplot as plt
    import pandas as pd
    # ahead plot
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,4))#
    #plt.figure(2)
    xaxis = ax.get_xaxis()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

    ax.plot(pd.date_range(startday_ahead, periods=len(data_val), freq='W'), data_val, color='k', linewidth=1)
    ax.plot(pd.date_range(startday_ahead, periods=len(pred_val), freq='W'), pred_val, linestyle='-', color='g', linewidth=1)
    plt.xlabel('Dates')
    plt.ylim([30, 140])
    ax.legend(['Data', 'Validation'], loc='upper left')
    plt.show()
    

def plot_fill_errors(data, predmins, predmaxs, predavgs, errors, startday, startdaypred):
    import pandas as pd
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,4))
    plt.figure(1)
    xaxis = ax.get_xaxis()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.fill_between(pd.date_range(startdaypred, periods=len(predmins), freq='W')[:-5], predmins[:-5], predmaxs[:-5],
                    facecolor='g')
    ax.plot(pd.date_range(startday, periods=len(data), freq='W')[:-5], data[:-5], 'k') 
    ax.set_xlabel('Dates')
    ax.set_ylabel(r'$^{222}$Rn ($Bq\cdot m^{-3}$)', fontsize=16)
    ax.legend(['Data', 'CNN (Rn + T)'], loc='upper left')
    ax.set_ylim([30, 140])
    ax.grid()

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
    axins = zoomed_inset_axes(ax, 1.7, loc='lower left', bbox_to_anchor=(673,223))
    axins.xaxis.set_major_locator(mdates.YearLocator())
    axins.xaxis.set_minor_locator(mdates.MonthLocator())
    axins.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axins.fill_between(pd.date_range(startdaypred, periods=len(predmins), freq='W')[:-5], predmins[:-5], predmaxs[:-5],
                       facecolor='g')
    axins.plot(pd.date_range(startday, periods=len(data), freq='W')[:-5], data[:-5], 'k') 
    axins.set_xlim('2017-08-15', '2019-06-24')
    axins.set_ylim(50, 110)
    axins.set_yticks([60, 80, 100])
    axins.set_title('a)', loc='right', x=0.98, y=0.85)
    axins.grid()

    axins2 = zoomed_inset_axes(ax, 1.7, loc='lower left', bbox_to_anchor=(673,97))
    axins2.xaxis.set_major_locator(mdates.YearLocator())
    axins2.xaxis.set_minor_locator(mdates.MonthLocator())
    axins2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axins2.vlines(pd.date_range(startdaypred, periods=len(predavgs), freq='W')[:-5], 0, errors)
    axins2.set_xlim('2017-08-15', '2019-06-24')
    axins2.set_yticks([int(np.min(errors)+10), 0, int(np.max(errors)+1)])
    axins2.set_ylim([-20, 20])
    axins2.grid()
    axins2.yaxis.set_label_position('right')
    axins2.yaxis.labelpad = 20
    axins2.set_ylabel('Y - pred', fontsize=13, rotation=270)
    axins2.set_title('b)', loc='right', x=0.98, y=0.85)

    plt.xticks(visible=True)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    plt.suptitle('Weekly Fitting at LSC - Hall A')
    plt.show()
    

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
        
def show_errors_v2(neurons, Xtrainlist, Y_train, Xtest_list, Y_test, arr_str, iterations, sample_size, DF_mdnRnA):
    for i in range(len(Xtrainlist)):
        print('\n\n#########\n', arr_str[i], '\n########\n\n')
        ECM = []
        EAM = []
        for it in range(iterations):
            #print('Iteration ', it)
            history, pred, acc_train, acc_test, model = NN_v2(neurons, 90, Xtrainlist[i], Y_train, Xtest_list[i], Y_test, sample_size)
            predmaxs, predmins, predavgs = extract_maxs_mins_avgs(pred)
            Y_test_error = DF_mdnRnA[DF_mdnRnA['dates'] > '2017-08-06']['mdnRnA']
            ECM.append(mean_squared_error(Y_test_error, predavgs))
            EAM.append(mean_absolute_error(Y_test_error, predavgs))
        print('ECM_'+arr_str[i]+' = ', ECM)
        print('EAM_'+arr_str[i]+' = ', EAM)

def Join_DF_RnT(*args):
    output = copy(args[0])
    for i in range(1, len(args)):
        output['tmed'+str(i)] = args[i]['tmed']
    return output