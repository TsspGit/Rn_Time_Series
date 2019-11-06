__author__ = '@Tssp'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats.stats import pearsonr
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from utils.NNutils import loadallDF
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=13
plt.rcParams['font.size']=16
plt.rcParams['xtick.major.pad']='8'
plt.rc('text',usetex=True)
plt.rc('font',family='serif')

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

# Plot:
X = BCN_arima[['tmed', 'velmedia', 'presmed', 'mdnRnA']]
Y = mdnRnA
print(X.head())

test_size=int(0.3 * len(mdnRnA))
Xtrain, Xtest = X[:-test_size], X[-test_size:]
Ytrain, Ytest = Y[:-test_size], Y[-test_size:]

clf = RandomForestRegressor(n_estimators=10, n_jobs=2, max_features='sqrt', random_state=42)
clf.fit(Xtrain, Ytrain)
pred = clf.predict(Xtest)
print(clf.score(Xtest, Ytest))

importances = np.sort(clf.feature_importances_)
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
plt.figure(figsize=(12, 8))
plt.bar(range(X.shape[1]), importances, color="dimgray", edgecolor='k', lw=3, align="center")
plt.xticks(range(X.shape[1]), [r'$\bar{V}$', r'$\bar{P}$', r'$\bar{T}$', r'$^{222}Rn$'])
plt.xlim([-1, X.shape[1]])
plt.ylim([0, 1])
#plt.xlabel('Features')
plt.ylabel('Relative Importance')
plt.savefig('../Figures/Variables_Importance.eps', dpi=100)
plt.savefig('../Figures/Variables_Importance.png', dpi=100)
