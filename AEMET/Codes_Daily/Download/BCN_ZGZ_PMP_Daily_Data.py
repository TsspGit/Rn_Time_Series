import numpy as np
import pandas as pd
import urllib
import json
import requests
import matplotlib.pyplot as plt

from AEMET_Daily_class import AEMET_GET

def main(years, IDs, names=''):
    api_key='eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0c3NhbmNoZXpwYXN0b3JAZ21haWwuY29tIiwianRpIjoiYzQzZGM1ZmYtNmRiYS00MzFmLTk3OTEtZWMzNGE3YjUzMDI3IiwiaXNzIjoiQUVNRVQiLCJpYXQiOjE1NjY0Nzc2OTgsInVzZXJJZCI6ImM0M2RjNWZmLTZkYmEtNDMxZi05NzkxLWVjMzRhN2I1MzAyNyIsInJvbGUiOiIifQ.bjmGdiW9vQ2ThnrLryvCxv2tad8XRDXA9zlcBQRg-U4'

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

    print(Data.isnull().sum())

    assert len(Data['fecha'].unique()) == len(Data['fecha'])/3, 'Hay datos repetidos'
    print('No hay ningun dato repetido')

    Data.to_csv('~/Rn_Weekly_NN/AEMET/Data/Daily/BCN_ZGZ_PMP_Daily' + str(years[0]) + '-' + str(years[-1]) + '.csv')
    return Data

years = [2012, 2013, 2014, 2015, 2016, 2017, 2018]
IDs = ['0076', '9263D', '9434'] 
names = ['BCN', 'PMP', 'ZGZ']

main(years, IDs, names)