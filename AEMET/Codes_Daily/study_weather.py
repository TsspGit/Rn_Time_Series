__author__ = '@Tssp'

from utils.aemepy import DF_subplots2x2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rc('text',usetex=True)
plt.rc('font',family='serif')

file = sys.argv[1] # Name of the complete csv. i.e. BCN_ZGZ_NVR_HSC_Daily2013-2019.csv
DF = pd.read_csv('~/CIEMAT/Rn_Weekly_NN/AEMET/Data/Daily/{}'.format(file),
                 usecols=['fecha', 'tmed', 'presMax', 'presMin', 'velmedia'])
# 0076 -> BCN, 9263D -> NAVARRA, 9434 -> ZGZ, 9898 -> HUESCA
DF['presmed'] = DF[['presMax', 'presMin']].mean(axis=1)
DF = DF.drop(['dir', 'presMax', 'presMin'], axis=1)
print('How many nulls has the dataframe?', DF.isnull().sum())

## split by station:
BCN = DF[DF['indicativo'] == '0076']
NVR = DF[DF['indicativo'] == '9263D']
ZGZ = DF[DF['indicativo'] == '9434']
HSC = DF[DF['indicativo'] == '9898']

# Plot:
arr_plts = [BCN, NVR, ZGZ, HSC]
arr_str = ['BCN', 'NVR', 'ZGZ', 'HSC']
## Temperature:
print('plotting T...')
plt.figure(1, figsize=(12, 8), dpi=300)
DF_subplots2x2(arr_plts, 'fecha', 'tmed', arr_str, ylabel=r'\bar{T}\ (^o C)', xlabel='Dates', save=True, v='T')
## Preassure:
print('plotting P...')
plt.figure(2, figsize=(12, 8), dpi=300)
DF_subplots2x2(arr_plts, 'fecha', 'presmed', arr_str, ylabel=r'\bar{P}\ (hPa)', xlabel='Dates', save=True, v='P', c='b')
# Wind velocity:
print('plotting V...')
plt.figure(3, figsize=(12, 8), dpi=300)
DF_subplots2x2(arr_plts, 'fecha', 'velmedia', arr_str, ylabel=r'\bar{V}\ (ms^{-1})', xlabel='Dates', save=True, v='V', c='r')
plt.show()
