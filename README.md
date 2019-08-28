# Rn Time Series Analysis:

_In this repository one can find the main codes and results of this part of the project_

## Download 🚀

_Just run: git clone https://github.com/TsspGit/Rn_Time_Series.git in Linux to download the complete project._

## Just to Know

- Most of the .py files are duplicated in .ipynb (jupyter notebooks). Notebooks are the best option to see both the code and the results.

- Seasonal_Decomposition_Rn.ipynb shows the STL decomposition and the seasonality of the signal obtained using a Fast Fourier Transform (FFT)

- CNN_weekly_Tomy.py applies a convolutional neural network to predict the levels of 222Rn the last year of measurements.

- AEMET folder contains two main blocks Daily and Monthly. Those codes are practicaly the same with the difference that is connected to different urls, one for the averages values per month and the other one with the average values per day (which is the one of interest). In both folders one could find a first attempt to access to the data and a folder named Download, that contains the codes to download the final CSV.
