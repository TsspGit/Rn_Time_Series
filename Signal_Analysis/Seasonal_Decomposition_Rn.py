#!/usr/bin/env python
# coding: utf-8
__author__ = 'Tomás Sánchez Sánchez-Pastor'
__date__ = 'August 2019'

"""
How to run the code:
$ python3 Seasonal_Decomposition_Rn.py <True/False> 
True: save the figures
False: dont save the figures
"""

# Imports:

import numpy as np
import math as mt
import matplotlib.pyplot as plt
import pandas as pd
import sys
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from scipy import fftpack
import global_functions as glb
plt.rc('text',usetex=True)
plt.rc('font',family='serif')

if len(sys.argv) == 2 and (bool(sys.argv[1]) == True or bool(sys.argv[1]) == False):

	def main(save_bool=bool(sys.argv[1])):
		# Weekly data:

		mdnRnA = glb.read_datatxt('/afs/ciemat.es/user/t/tomas/CIEMAT/Rn_Weekly_NN/mdnRnA.txt', ',')
		# Recording dates:

		startday = pd.datetime(2013, 7, 1)
		dates = pd.date_range(startday, periods=len(mdnRnA), freq='W')

		# Creating dataset:

		series = pd.Series(data=mdnRnA, index=dates)
		print("Data:\n", series.head())

		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
		plt.figure(1)
		series.plot(linestyle='-', lw=2, color='purple')
		plt.xlabel('Time', fontsize=14)
		plt.ylabel('$^{222}$Rn($Bq \cdot m^{-3}$)', fontsize=14)
		plt.title('Weekly fitting at LSC - Hall A', fontsize=14)

		# STL Decomposition of the signal:

		decomp = seasonal_decompose(series, model='multiplicative')
		observed = decomp.observed
		trend = decomp.trend
		seasonal = decomp.seasonal
		residual = decomp.resid

		arr_plts = [observed, trend, seasonal, residual]
		arr_str = ['Observed', 'Trend', 'Seasonal', 'Residual']

		# Figure with the differents contributions:

		plt.figure(2, figsize=(12, 8))

		for i in range(len(arr_plts)):
		    ax = plt.subplot(2, 2, i+1)
		    arr_plts[i].plot(lw=1.5, color='purple')
		    plt.title('{}'.format(arr_str[i]), fontsize=13)
		    plt.grid(True)
		    if i == 0 or i == 2:
		        plt.ylabel('$^{222}$Rn($Bq \cdot m^{-3}$)', fontsize=13)
		        plt.xlabel('Years', fontsize=13)
		    plt.tight_layout()

		if save_bool:
			plt.savefig('./Rn_Seasonal_Decomposition.eps', bbox_inches='tight')


		# Fast Fourier Transform (FFT):

		FFT = np.abs(fftpack.fft(mdnRnA))[1:] # Drop the first value that explodes
		freqs = fftpack.fftfreq(len(mdnRnA), d=1)[1:]
		# Positive freqs
		i = (freqs >= 0)
		freqs = freqs[i]
		FFT = FFT[i]
		# Figure:

		plt.figure(3, figsize=(16, 8))

		ax = plt.subplot(2, 2, 1)
		observed.plot(color='purple', lw=1.5)
		plt.xlabel('Years', fontsize=14)
		plt.ylabel('$^{222}$Rn($Bq \cdot m^{-3}$)', fontsize=14)
		plt.grid(True)

		ax = plt.subplot(2, 2, 3)
		plt.plot(freqs, FFT, color='darkgreen', lw=1.5)
		plt.xlabel('Freq $(weeks^{-1})$', fontsize=14)
		plt.ylabel('FFT $^{222}$Rn($Bq \cdot Hz^{-1} \cdot m^{-3}$)', fontsize=14)
		plt.grid(True)

		plt.tight_layout()
		if save_bool:
			plt.savefig('./Rn_FFT.eps', bbox_inches='tight')

		j = (FFT == np.max(FFT))
		print("The most relevant period in the time series is: {} months".format(round(1/freqs[j][0], 2)))

		plt.show()

	main()

elif len(sys.argv) is not 2:
	print("You have to introduce only one parameter")
elif (bool(sys.argv[1]) is not True and bool(sys.argv[1]) is not False):
	print("Introduce True if you want to save the figures or False if not")
