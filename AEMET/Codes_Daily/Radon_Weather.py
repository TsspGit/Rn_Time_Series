__author__ = '@Tssp'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from utils.aemepy import Rn_Clima_subplots, Rn_Clima_plot
from scipy.stats.stats import pearsonr
plt.rc('text',usetex=True)
plt.rc('font',family='serif')
plt.rcParams['xtick.labelsize']=13
plt.rcParams['ytick.labelsize']=13

# Correlation print function:
def print_corr(DF_list, field):
    for i in range(len(DF_list)):
        try:
            print('\nThe correlation with the avg {} in {} is: '.format(field, arr_str[i]),
             '\nAveraged DF:\n',
                  pearsonr(DF_list[i][field].values, mdnRnA)[0])
        except:
            continue

# Load Cities:
list_cities = ['BCN', 'NVR', 'HSC', 'ZGZ']
weekly = {}
for city in list_cities:
    weekly[city] = pd.read_csv(f'../Data/Daily/{city}/{city}_weekly.csv',
                            usecols=['fecha', 'tmed', 'presmed', 'velmedia'])
    weekly[city + '_avg'] = pd.read_csv(f'../Data/Daily/{city}/{city}_avg_weekly.csv',
                                     usecols=['fecha', 'tmed', 'presmed', 'velmedia'])
    weekly[city + '_arima'] = pd.read_csv(f'../Data/Daily/{city}/{city}_arima_weekly.csv',
                                     usecols=['fecha', 'tmed', 'presmed', 'velmedia'])
    
# Load Rn:
mdnRnA = np.loadtxt('../../mdnRnA.txt', delimiter=',')
startday = pd.datetime(2013, 7, 1)
dates = pd.date_range(startday, periods=len(mdnRnA), freq='W')
assert len(mdnRnA) == len(weekly['BCN']['fecha'].values)
assert len(mdnRnA) == len(weekly['NVR']['fecha'].values)
assert len(mdnRnA) == len(weekly['HSC']['fecha'].values)
assert len(mdnRnA) == len(weekly['ZGZ']['fecha'].values)


############################################################
# Not nulls Datasets:                                      #
############################################################
BCN = weekly['BCN']
BCN['fecha'] = pd.to_datetime(BCN['fecha'])
NVR = weekly['NVR']
NVR['fecha'] = pd.to_datetime(NVR['fecha'])
HSC = weekly['HSC']
HSC['fecha'] = pd.to_datetime(HSC['fecha'])
ZGZ = weekly['ZGZ']
ZGZ['fecha'] = pd.to_datetime(ZGZ['fecha'])
DF_list = [BCN, NVR, ZGZ, HSC]
arr_str = ['BCN', 'PMP', 'ZGZ', 'HSC']

## 4 cities in a plot:
### Tempeature:
plt.figure(1, figsize=(14, 10), dpi=300)
Rn_Clima_subplots(DF_list, mdnRnA, dates, 'tmed', arr_str, ylabel=r'\bar{T}\ (^o C)', c='#1f77b4', legend=r'\bar{T}',
                  v='T',
 save=True)
### Preassure
plt.figure(2, figsize=(14, 10), dpi=300)
Rn_Clima_subplots(DF_list, mdnRnA, dates, 'presmed', arr_str, ylabel=r'\bar{P}\ (hPa)', c='#2ca02c', legend=r'\bar{P}',
                   v='P',
 save=True)
### Wind velocity
plt.figure(3, figsize=(14, 10), dpi=300)
Rn_Clima_subplots(DF_list, mdnRnA, dates, 'velmedia', arr_str, ylabel=r'\bar{V}\ (m/s)', c='#d62728', legend=r'\bar{V}',
                   v='V',
 save=True)

# 1 plot per city:
## Temperature:
Rn_Clima_plot(DF_list, mdnRnA, dates, 'tmed', arr_str, ylabel=r'\bar{T}\ (^o C)', c='#1f77b4', legend=r'\bar{T}',
                   v='T',
 save=True)
### Preassure:
Rn_Clima_plot(DF_list, mdnRnA, dates, 'presmed', arr_str, ylabel=r'\bar{P}\ (hPa)', c='#2ca02c', legend=r'\bar{P}',
                   v='P',
 save=True)
### Wind velocity:
Rn_Clima_plot(DF_list, mdnRnA, dates, 'velmedia', arr_str, ylabel=r'\bar{V}\ (m/s)', c='#d62728', legend=r'\bar{V}',
                   v='V',
 save=True)


############################################################
# Averaged Datasets:                                       #
############################################################
BCN_avg = weekly['BCN_avg']
BCN_avg['fecha'] = pd.to_datetime(BCN_avg['fecha'])
NVR_avg = weekly['NVR_avg']
NVR_avg['fecha'] = pd.to_datetime(NVR_avg['fecha'])
HSC_avg = weekly['HSC_avg']
HSC_avg['fecha'] = pd.to_datetime(HSC_avg['fecha'])
ZGZ_avg = weekly['ZGZ_avg']
ZGZ_avg['fecha'] = pd.to_datetime(ZGZ_avg['fecha'])
DFavg_list = [BCN_avg, NVR_avg, ZGZ_avg, HSC_avg]
arravg_str = ['BCN_avg', 'PMP_avg', 'ZGZ_avg', 'HSC_avg']

## 4 cities in a plot:
### Tempeature:
plt.figure(16, figsize=(14, 10), dpi=300)
Rn_Clima_subplots(DFavg_list, mdnRnA, dates, 'tmed', arr_str, ylabel=r'\bar{T}\ (^o C)', c='#1f77b4', legend=r'\bar{T}',
 v='Tavg', save=True)
### Preassure
plt.figure(17, figsize=(14, 10), dpi=300)
Rn_Clima_subplots(DFavg_list, mdnRnA, dates, 'presmed', arr_str, ylabel=r'\bar{P}\ (hPa)', c='#2ca02c',legend=r'\bar{P}',
 v='Pavg', save=True)
### Wind velocity
plt.figure(18, figsize=(14, 10), dpi=300)
Rn_Clima_subplots(DFavg_list, mdnRnA, dates, 'velmedia', arr_str, ylabel=r'\bar{V}\ (m/s)', c='#d62728',legend=r'\bar{V}',
 v='Vavg', save=True)

## 1 plot per city:
### Temperature:
Rn_Clima_plot(DFavg_list, mdnRnA, dates, 'tmed', arr_str, ylabel=r'\bar{T}\ (^o C)', c='#1f77b4',legend=r'\bar{T}',
 v='Tavg', save=True)
### Preassure:
Rn_Clima_plot(DFavg_list, mdnRnA, dates, 'presmed', arr_str, ylabel=r'\bar{P}\ (hPa)', c='#2ca02c',legend=r'\bar{P}',
 v='Pavg', save=True)
### Wind velocity:
Rn_Clima_plot(DFavg_list, mdnRnA, dates, 'velmedia', arr_str, ylabel=r'\bar{V}\ (m/s)', c='#d62728',legend=r'\bar{V}',
 v='Vavg', save=True)

############################################################
# ARIMA Datasets:                                          #
############################################################
BCN_arima = weekly['BCN_arima']
BCN_arima['fecha'] = pd.to_datetime(BCN_arima['fecha'])
NVR_arima = weekly['NVR_arima']
NVR_arima['fecha'] = pd.to_datetime(NVR_arima['fecha'])
HSC_arima = weekly['HSC_arima']
HSC_arima['fecha'] = pd.to_datetime(HSC_arima['fecha'])
ZGZ_arima = weekly['ZGZ_arima']
ZGZ_arima['fecha'] = pd.to_datetime(ZGZ_arima['fecha'])
DFarima_list = [BCN_arima, NVR_arima, ZGZ_arima, HSC_arima]
arrarima_str = ['BCN_arima', 'PMP_arima', 'ZGZ_arima', 'HSC_arima']

## 4 cities in a plot:
### Tempeature:
plt.figure(31, figsize=(14, 10), dpi=200)
Rn_Clima_subplots(DFarima_list, mdnRnA, dates, 'tmed', arr_str, ylabel=r'\bar{T}\ (^o C)', c='#1f77b4',legend=r'\bar{T}',
 v='Tarima', save=True)
### Preassure
plt.figure(32, figsize=(14, 10), dpi=200)
Rn_Clima_subplots(DFarima_list, mdnRnA, dates, 'presmed', arr_str, ylabel=r'\bar{P}\ (hPa)', c='#2ca02c',legend=r'\bar{P}',
 v='Parima', save=True)
### Wind velocity
plt.figure(33, figsize=(14, 10), dpi=200)
Rn_Clima_subplots(DFarima_list, mdnRnA, dates, 'velmedia', arr_str, ylabel=r'\bar{V}\ (m/s)', c='#d62728',legend=r'\bar{V}',
 v='Varima', save=True)

## 1 plot per city:
### Temperature:
Rn_Clima_plot(DFarima_list, mdnRnA, dates, 'tmed', arr_str, ylabel=r'\bar{T}\ (^o C)', c='#1f77b4',legend=r'\bar{T}',
 v='Tarima', save=True)
### Preassure:
Rn_Clima_plot(DFarima_list, mdnRnA, dates, 'presmed', arr_str, ylabel=r'\bar{P}\ (hPa)', c='#2ca02c',legend=r'\bar{P}',
 v='Parima', save=True)
### Wind velocity:
Rn_Clima_plot(DFarima_list, mdnRnA, dates, 'velmedia', arr_str, ylabel=r'\bar{V}\ (m/s)', c='#d62728',legend=r'\bar{V}',
 v='Varima', save=True)

# Print correlations:
print_corr(DF_list, 'tmed')
print_corr(DFavg_list, 'tmed')
print_corr(DFarima_list, 'tmed')
print_corr(DF_list, 'presmed')
print_corr(DFavg_list, 'presmed')
print_corr(DFarima_list, 'presmed')
print_corr(DF_list, 'velmedia')
print_corr(DFavg_list, 'velmedia')
print_corr(DFarima_list, 'velmedia')