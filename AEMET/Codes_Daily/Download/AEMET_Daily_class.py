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
    
    def connect(self, api_key, where, ID=''):
        if where == 'Data':
            url = 'https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{}-01-01T00:00:00UTC/fechafin/{}-12-31T23:59:59UTC/estacion/{}/?api_key={}'.format(self.start, self.end, ID, api_key)
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
    
    def print_metadata(self, js, fields_m):
        metadata = json.loads(urllib.request.urlopen(js['metadatos']).read().decode("iso8859-1"))
        def show_metadata(fields_m):
            for dic in range(len(metadata['campos'])):
                if metadata['campos'][dic]['id'] in fields_m:
                    print(metadata['campos'][dic], '\n')
        show_metadata(fields_m)
        
    def get_DF(self, fields, js):
        data = json.loads(urllib.request.urlopen(js['datos']).read().decode())
        DF = pd.DataFrame(data)
        DF = DF.filter(fields)
        # Replace , to .
        def replace_comas(field):
            output = []
            for elem in field.values:
                if isinstance(elem, str) and',' in elem:
                    output.append(elem.replace(',', '.'))
                else:
                    output.append(elem)
            return output   
        tmed = replace_comas(DF['tmed'])
        velmedia = replace_comas(DF['velmedia'])
        presMax = replace_comas(DF['presMax'])
        presMin = replace_comas(DF['presMin'])
        DF['tmed'] = tmed
        DF['velmedia'] = velmedia
        DF['presMax'] = presMax
        DF['presMin'] = presMin
        # Cast the fields
        DF['tmed'] = DF['tmed'].astype(float)
        DF['dir'] = DF['dir'].astype(float)
        DF['velmedia'] = DF['velmedia'].astype(float) 
        DF['presMax'] = DF['presMax'].astype(float) 
        DF['presMin'] = DF['presMin'].astype(float) 
        return DF
    
    def get_stations(self, airports, js_stations):
        data_stations = json.loads(urllib.request.urlopen(js_stations['datos']).read().decode('latin-1'))
        DF_st = pd.DataFrame(data_stations)
        DF_st = DF_st.filter(['indicativo', 'latitud', 'longitud', 'nombre', 'provincia'])
        airports_position = DF_st['nombre'].str.contains('AEROPUERTO')
        DF_airst = DF_st[airports_position]
        def Final_DF(airports=None):
            # inputs a list with the names of the airport's cities
            i = 0
            output = {}
            for elem in airports:
                i += 1
                pos = DF_airst['nombre'].str.contains(elem)
                DF_final = DF_airst[pos]
                output['station' + str(i)] =  DF_final
            return output
        DF_airports_dic = Final_DF(airports)
        def concat_dictionary(DIC):
            # Inputs a dictionary with the dataframes
            keys_list = [key for key in DIC.keys()]
            for j in range(len(keys_list)):
                if j == 0:
                    Final = DIC[keys_list[0]]
                else:
                    Final = pd.concat([Final, DIC[keys_list[j]]])
            return Final
        DF_airports = concat_dictionary(DF_airports_dic)
        return DF_airports