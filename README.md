# Rn Time Series Analysis: 
<img align="right" src="simbolo_radon.png">

_In this repository one can find the main codes and results of this part of the project_

## Download 🚀

_Just run: git clone git@github.com:TsspGit/Rn_Time_Series.git in Linux to download the complete project._

## Just to Know

- Most of the .py files are duplicated in .ipynb (jupyter notebooks). Notebooks are the best option to see both the code and the results.

## Signal_Analysis folder

1. **Seasonal_Decomposition_Rn.py/ipynb** shows the STL decomposition and the seasonality obtained using a FFT.
How to run Seasonal_Decomposition_Rn.py?
```
$ python3 Seasonal_Decomposition_Rn.py <True/False>
True: save the resulting plots
False: doesn't save the resulting plots
```

2. **CNN_weekly_Tomy.py/ipynb** applies a convolutional neural network to forecasting the Radon density levels.

3. In **Rn_Time_Series.pdf** one can find a summary of the signal analysis and the most relevant results.

## AEMET folder

### Codes_Daily

1. **Download** contains the codes for download the data from the AEMET api, **it is required to own an api key** (https://opendata.aemet.es/centrodedescargas/altaUsuario):
	- **AEMET_Daily_Test.ipynb** is the first attempt to access the data and the information of the stations.
	- **AEMET_Daily_class.py** is the brain. To see how it works see the notebook version
	- **Download_Airports_st.py** is the code used to obtain the final dataframe. The inputs are the cities of the airports stations and the years. The code automatically relates the name of the city with the station ID and append the collumns to a dataframe. Finally, filters the duplicates, sort by date and save it on a .csv file. See the notebook for further details.
	- **study_weather.py** takes the dataframe of the city stations downloaded before and plots the average temperature, preassure and wind velocity for each date. An example of running this code on a linux terminal is: *python3 study_weather.py BCN_ZGZ_NVR_HSC_Daily2013-2019.csv*
	- **split_by_city.py** saves three csv's per city: the complete csv, skipped rows with missing values csv and filled missing values with the avg per month of the correct collumn.
	- **weekly_Data.py** takes the skipped rows with missing values csv and filled missing values with the avg per month csv as input to create two new csvs with the avg values per week.

2. **First Data Handling.ipynb** is the first attemp to plot the data.

### Codes_Monthly

Follows the same logic than **Codes_Daily**, the difference is that here we access to the averages per month, and this is not useful for carry out our study.

### Data

The target folder with the final csv.
