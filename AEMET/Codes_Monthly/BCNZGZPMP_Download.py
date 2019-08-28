import numpy as np
import pandas as pd
import urllib
import json
import requests
from AEMET_class import AEMET_GET

years = [2012, 2013, 2014, 2015, 2016, 2017, 2018]
IDs = ['0076', '9263D', '9434'] 
names = ['BCN', 'PMP', 'ZGZ']
api_key='eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0c3NhbmNoZXpwYXN0b3JAZ21haWwuY29tIiwianRpIjoiYzQzZGM1ZmYtNmRiYS00MzFmLTk3OTEtZWMzNGE3YjUzMDI3IiwiaXNzIjoiQUVNRVQiLCJpYXQiOjE1NjY0Nzc2OTgsInVzZXJJZCI6ImM0M2RjNWZmLTZkYmEtNDMxZi05NzkxLWVjMzRhN2I1MzAyNyIsInJvbGUiOiIifQ.bjmGdiW9vQ2ThnrLryvCxv2tad8XRDXA9zlcBQRg-U4'

i = 0
for ID in IDs:
    name = names[i]
    i += 1
    print('\n\nStation: '+ name)
    for year in years:
        print('Year: {}:'.format(year))
        A = AEMET_GET(year, year)
        js = A.connect(api_key, 'Data', ID)
        DF = A.get_DF(['fecha', 'tm_mes', 'q_med', 'w_racha'], js)
        print(DF.head())
        DF.to_csv('~/Rn_Weekly_NN/AEMET/Data/{}'.format(name) + str(year) + '.csv')