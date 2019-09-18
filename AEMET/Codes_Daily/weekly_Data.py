__author__ = '@Tssp'

import pandas as pd
from utils.aemepy import avg_per_weeks

# Loading:
list_cities = ['BCN', 'NVR', 'HSC', 'ZGZ']
dic = {}
for city in list_cities:
    dic[city] = pd.read_csv(f'~/CIEMAT/Rn_Weekly_NN/AEMET/Data/Daily/{city}/{city}_notnulls.csv',
                            usecols=['fecha', 'tmed', 'presmed', 'velmedia'])
    dic[city + '_avg'] = pd.read_csv(f'~/CIEMAT/Rn_Weekly_NN/AEMET/Data/Daily/{city}/{city}_avgfilled.csv',
                                     usecols=['fecha', 'tmed', 'presmed', 'velmedia'])

# New DF with means by weeks
output = {}
for key in dic.keys():
    output[key + '_weekly'] = avg_per_weeks(dic[key])

# Saving:
for key in output.keys():
    output[key].to_csv(f'~/CIEMAT/Rn_Weekly_NN/AEMET/Data/Daily/{key[:3]}/{key}.csv')