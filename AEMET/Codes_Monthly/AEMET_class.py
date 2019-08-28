import numpy as np
import pandas as pd
import urllib
import json
import requests

class AEMET_GET:
    """This class is developed with the proposal of download the data from several stations at a chosen year. 
    Inputs: Starting and ending date with format AAAA
    Methods: connect -> Inputs the api key, an string indicating what information you want to access, Data or Station
             print_metadata -> Inputs the Json returned by connect and the fields you want to know and print the
                            information
             get_DF -> Inputs the fields that you want to stack to the DF and the json of the Data connection 
             get_stations -> Inputs the cities of the airports that you want to download and
             the  Json Station connection
    """
    
    def __init__(self, start_date, end_date):
        self.start = str(start_date)
        self.end = str(end_date)
        print("""\n You are going to Download the data from {} to {} 
from the stations with IDs:\n""".format(start_date, end_date))
        for ID in ['0076', '9263D', '9434']:
            print(ID)
    
    def connect(self, api_key, where, ID):
        if where == 'Data':
            url = 'https://opendata.aemet.es/opendata/api/valores/climatologicos/mensualesanuales/datos/anioini/{}/aniofin/{}/estacion/{}/?api_key={}'.format(self.start, self.end, ID, api_key)
            api_json = urllib.request.urlopen(url).read().decode()
            js = json.loads(api_json)
            print(api_json)
            return js
        elif where == 'Station':
            url_estaciones = 'https://opendata.aemet.es/opendata/api/valores/climatologicos/inventarioestaciones/todasestaciones/?api_key={}'.format(api_key)
            api_stations = urllib.request.urlopen(url_estaciones).read().decode()
            js_stations = json.loads(api_stations)
            print(api_stations)
            return js_stations
        else:
            raise TypeError('ERROR! Possible conections: Data OR Station')
    
    def print_metadata(self, js, fields):
        metadata = json.loads(urllib.request.urlopen(js['metadatos']).read().decode("iso8859-1"))
        def show_metadata(fields):
            for dic in range(len(metadata['campos'])):
                if metadata['campos'][dic]['id'] in fields:
                    print(metadata['campos'][dic], '\n')
        show_metadata(fields)
        
    def get_DF(self, fields, js):
        data = json.loads(urllib.request.urlopen(js['datos']).read().decode())
        data = data[:12]
        DF = pd.DataFrame(data)
        DF = DF.filter(fields)
        def w_racha_splitter():
            w_racha = DF['w_racha'].values
            direction = []
            V = []
            for elem in w_racha:
                if isinstance(elem, str):
                    direction.append(elem[:-4].split('/')[0])
                    V.append(elem[:-4].split('/')[1])
                else: 
                    direction.append(elem)
                    V.append(elem)
            return direction, V
        direction, V = w_racha_splitter()
        DF['w_direction'] = direction
        DF['Vel'] = V
        # Cast the fields
        DF['tm_mes'] = DF['tm_mes'].astype(float)
        DF['q_med'] = DF['q_med'].astype(float)
        DF['w_direction'] = DF['w_direction'].astype(float) 
        DF['Vel'] = DF['Vel'].astype(float) 
        DF = DF.drop(columns = 'w_racha')
        return DF
    
    def get_stations(self, airports, js_stations):
        data_stations = json.loads(urllib.request.urlopen(js_stations['datos']).read().decode('latin-1'))
        DF_st = pd.DataFrame(data_stations)
        DF_st = DF_st.filter(['indicativo', 'latitud', 'longitud', 'nombre', 'provincia'])
        airports_position = DF_st['nombre'].str.contains('AEROPUERTO')
        DF_airst = DF_st[airports_position]
        def Final_DF(airports=None):
            i = 0
            output = {}
            for elem in airports:
                i += 1
                pos = DF_airst['nombre'].str.contains(elem)
                DF_final = DF_airst[pos]
                output['station' + str(i)] =  DF_final
            return pd.concat([output['station1'], output['station2'], output['station3']])
        DF_airports = Final_DF(['BARCELONA', 'PAMPLONA', 'ZARAGOZA'])
        return DF_airports