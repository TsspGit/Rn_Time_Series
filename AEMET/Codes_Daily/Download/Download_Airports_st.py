__author__ = '@Tssp'

import numpy as np
import pandas as pd
import urllib
import json
import requests
import matplotlib.pyplot as plt

from AEMET_Daily_class import AEMET_GET

def main(years=None, cities=None):
    api_key='eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0c3NhbmNoZXpwYXN0b3JAZ21haWwuY29tIiwianRpIjoiYzQzZGM1ZmYtNmRiYS00MzFmLTk3OTEtZWMzNGE3YjUzMDI3IiwiaXNzIjoiQUVNRVQiLCJpYXQiOjE1NjY0Nzc2OTgsInVzZXJJZCI6ImM0M2RjNWZmLTZkYmEtNDMxZi05NzkxLWVjMzRhN2I1MzAyNyIsInJvbGUiOiIifQ.bjmGdiW9vQ2ThnrLryvCxv2tad8XRDXA9zlcBQRg-U4'
    
    #Obtain Ids of the airports
    A = AEMET_GET(2012, 2012)
    js_stations = A.connect(api_key, 'Station')
    DF_air = A.get_stations(cities, js_stations)
    print(DF_air)
    IDs, names = DF_air['indicativo'].values, DF_air['provincia'].values
    
    def Complete_Dictionary(names, years, IDs, fields):
        i = 0
        DIC = {}
        for ID in IDs:
            name = names[i]
            i += 1
            print('\n\nStation: '+ name)
            for year in years:
                print('Year: {}:'.format(year))
                A = AEMET_GET(year, year)
                js = A.connect(api_key, 'Data', ID)
                DIC[ID + str(year)] = A.get_DF(fields, js)
        return DIC

    fields = ['indicativo', 'fecha', 'tmed', 'dir', 'velmedia', 'presMax', 'presMin']
    DIC = Complete_Dictionary(names, years, IDs, fields)

    def concat_dictionary(DIC):
        keys_list = [key for key in DIC.keys()]
        for j in range(len(keys_list)):
            if j == 0:
                Final = DIC[keys_list[0]]
            else:
                Final = pd.concat([Final, DIC[keys_list[j]]])
        return Final

    Data = concat_dictionary(DIC)
    
    print("How many nulls are in the dataset? ", Data.isnull().sum())
    
    def filter_duplicates(DF):
        duplicated = DF.duplicated()
        filter_duplicates = DF.duplicated(keep='first')
        return Data[~filter_duplicates]
    
    Data = filter_duplicates(Data)

    Data.sort_values('fecha').to_csv('~/CIEMAT/Rn_Weekly_NN/AEMET/Data/Daily/BCN_ZGZ_NVR_HSC_Daily' + str(years[0]) + '-' + str(years[-1]) + '.csv')
    return Data

years = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
cities = ['BARCELONA', 'ZARAGOZA', 'PAMPLONA', 'HUESCA']

main(years, cities)