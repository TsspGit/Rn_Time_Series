__author__ = '@Tssp'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from utils.NNutils import loadallDF

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

# Random Forest:
Y = mdnRnA
test_size=int(0.3 * len(mdnRnA))
var = ['velmedia', 'presmed', 'tmed', 'mdnRnA']

for i in range(len(DF_list)):
	X = DF_list[i][['tmed', 'velmedia', 'presmed', 'mdnRnA']]
	Xtrain, Xtest = X[:-test_size], X[-test_size:]
	Ytrain, Ytest = Y[:-test_size], Y[-test_size:]
	clf = RandomForestRegressor(n_estimators=10, n_jobs=2, max_features='sqrt', random_state=42)
	clf.fit(Xtrain, Ytrain)
	pred = clf.predict(Xtest)
	importances = np.sort(clf.feature_importances_)
	print("##########", arr_str[i], "##########\n")
	for j in range(len(importances)):
		print(var[j], '\n', importances[j], "\n")