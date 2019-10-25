__author__ = '@Tssp'

import pandas as pd
import numpy as np
import sys
from utils.aemepy import fill_avg_per_month, fill_arima, avg_per_weeks
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd

file = sys.argv[1] # Name of the complete csv. i.e. BCN_ZGZ_NVR_HSC_Daily2013-2019.csv
DF = pd.read_csv('~/CIEMAT/Rn_Weekly_NN/AEMET/Data/Daily/{}'.format(file),
                 usecols=['fecha', 'tmed'])

# fill with ARIMA over avg per month
out = fill_arima(DF, ['tmed'])
out.to_csv('../Data/Daily/' + 'Validation_arima.csv')

dic = {'val': out}

# New DF with means by weeks
output = {}
for key in dic.keys():
    output[key + '_weekly'] = avg_per_weeks(dic[key])

# Saving:
for key in output.keys():
    output[key].to_csv(f'~/CIEMAT/Rn_Weekly_NN/AEMET/Data/Daily/val_weekly.csv')

print('\n###########\nDone\n############\n')