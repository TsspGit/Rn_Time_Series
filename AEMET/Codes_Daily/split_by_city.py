__author__ = '@Tssp'

import pandas as pd
import numpy as np
import sys
from utils.aemepy import fill_avg_per_month

file = sys.argv[1] # Name of the complete csv
DF = pd.read_csv('~/CIEMAT/Rn_Weekly_NN/AEMET/Data/Daily/{}'.format(file), usecols=range(1, 8))
BCN = DF[DF['indicativo'] == '0076']
NVR = DF[DF['indicativo'] == '9263D']
ZGZ = DF[DF['indicativo'] == '9434']
HSC = DF[DF['indicativo'] == '9898']

# with na
BCN[(BCN['fecha'] >= '2013-07-07') & (BCN['fecha'] <= '2019-07-21')].sort_values(['fecha']).to_csv('../Data/Daily/BCN/' + 'BCN.csv')
NVR[(NVR['fecha'] >= '2013-07-07') & (NVR['fecha'] <= '2019-07-21')].sort_values(['fecha']).to_csv('../Data/Daily/NVR/' + 'NVR.csv')
HSC[(HSC['fecha'] >= '2013-07-07') & (HSC['fecha'] <= '2019-07-21')].sort_values(['fecha']).to_csv('../Data/Daily/HSC/' + 'HSC.csv')
ZGZ[(ZGZ['fecha'] >= '2013-07-07') & (ZGZ['fecha'] <= '2019-07-21')].sort_values(['fecha']).to_csv('../Data/Daily/ZGZ/' + 'ZGZ.csv')

# dropna
BCN_dropna = BCN[(BCN['fecha'] >= '2013-07-07') & (BCN['fecha'] <= '2019-07-21')].sort_values(['fecha']).dropna()
BCN_dropna.to_csv('../Data/Daily/BCN/' + 'BCN_notnulls.csv')

NVR_dropna = NVR[(NVR['fecha'] >= '2013-07-07') & (NVR['fecha'] <= '2019-07-21')].sort_values(['fecha']).dropna()
NVR_dropna.to_csv('../Data/Daily/NVR/' + 'NVR_notnulls.csv')

HSC_dropna = HSC[(HSC['fecha'] >= '2013-07-07') & (HSC['fecha'] <= '2019-07-21')].sort_values(['fecha']).dropna()
HSC_dropna.to_csv('../Data/Daily/HSC/' + 'HSC_notnulls.csv')

ZGZ_dropna = ZGZ[(ZGZ['fecha'] >= '2013-07-07') & (ZGZ['fecha'] <= '2019-07-21')].sort_values(['fecha']).dropna()
ZGZ_dropna.to_csv('../Data/Daily/ZGZ/' + 'ZGZ_notnulls.csv')

# fill with avg per month:
BCN_fillavg = fill_avg_per_month(BCN)
BCN_fillavg.to_csv('../Data/Daily/BCN/' + 'BCN_avgfilled.csv')

NVR_fillavg = fill_avg_per_month(NVR)
NVR_fillavg.to_csv('../Data/Daily/NVR/' + 'NVR_avgfilled.csv')

HSC_fillavg = fill_avg_per_month(HSC)
HSC_fillavg.to_csv('../Data/Daily/HSC/' + 'HSC_avgfilled.csv')

ZGZ_fillavg = fill_avg_per_month(ZGZ)
ZGZ_fillavg.to_csv('../Data/Daily/ZGZ/' + 'ZGZ_avgfilled.csv')
print('###########\nDone\n############')