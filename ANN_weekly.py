#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

__author__ = ' AUTOR: Miguel Cardenas-Montes'

''' RNN Bidireccional en Keras '''


"""Imports: librerias"""
import os
import math
from math import sqrt
import sys
import numpy
import numpy as np
import random
import scipy.stats
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from time import time

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


import pandas as pd  

import tensorflow as tf

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates


##Para cambiar las fuentes
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'large'
mpl.rcParams['axes.labelsize'] = 'large'

import matplotlib as mpl
np.set_printoptions(threshold=np.inf)

import pdb # Para depurar
import copy


##########################################
############## Codigo ####################
##########################################

"""Codigo principal"""
def main():


	"""
	# Medians Rn222 HallA
	mdnRnA=[87.0, 88.0, 112.0, 99.0, 102.0, 86.0, 100.0, 88.0, 75.0, 94.0 ,80.0, 86.0, 76.0, 75.0 ,78.0, 67.0 ,67.0, 79.0 ,54.0, 66.0, 63.0, 60.5, 54.0, 63.0, 65.0, 73.0, 84.0, 70.0, 72.5, 77.0, 74.0, 84.0, 81.0, 81.5, 97.0, 97.0, 92.0, 80.0, 87.0, 90.0, 74.0, 75.0, 78.0, 82.0, 78.0, 75.0, 67.0, 74.0, 89.0, 86.0, 81.0, 85.0, 84.0, 81.0, 79.0, 99.0, 114.0, 94.0, 64.0, 76.0, 69.0, 74.0, 71.0, 79.0, 68.0, 64.0, 62.0, 67.0, 60.0, 64.0, 73.0, 68.0, 60.0, 72.0, 71.0, 71.0, 68.0, 77.0, 67.0, 46.0, 75.0, 86.0, 80.0, 88.0, 85.0, 61.0, 80.0, 98.0, 108.0, 89.0, 68.0, 65.0, 91.0,96.0, 106.5, 108.0, 106.5, 109.5, 104.0, 110.0, 86.0, 85.0, 89.0, 85.0, 71.0, 82.5, 75.0, 83.0, 73.0, 57.0, 55.0, 60.0, 63.0, 66.0, 79.0, 74.0, 81.0, 72.5, 67.0, 79.0, 73.0, 73.0, 70.0, 78.5, 69.0, 69.0, 65.0, 75.0, 72.0, 71.0, 72.5, 82.0, 100.0, 97.0, 97.0, 92.0, 101.0, 102.0, 73.0, 69.0, 72.0, 66.5, 72.0, 65.5, 75.5, 73.0, 63.0, 89.0, 98.0, 69.0 , 84.0, 81.0, 77.0, 65.0, 70.0, 87.0, 74.0, 66.0, 80.0, 69.0, 66.0, 62.0, 67.0, 65.0, 74.0, 77.0, 81.0, 79.0, 65.0, 72.0, 81.0, 70.0, 70.0, 70.0, 79.0, 81.0, 91.0, 84.0,73.5, 99.0, 101.0, 67.0, 76.0, 70.0, 74.0, 73.0, 80.0, 77.0, 104.0, 74.0, 80.0, 80.0, 86.0, 105.0, 97.0, 93.5, 92.0,76.0, 68.0, 56.0, 59.0, 72.0, 68.0, 79.0, 64.0, 71.5, 76.0, 74.0, 59.0, 74.0, 64.0, 68.0, 64.0, 67.0, 70.0, 77.0, 67.0, 71.0, 80.0, 84.0, 83.0, 70.0, 70.0, 95.0, 102.0, 109.0, 88.0, 89.0]
	print(len(mdnRnA))
	"""

	# Medians Rn222 HallA (correcto 52 semanas X 5 años =259 valores)
	mdnRnA=[90, 79, 99, 117, 99, 99, 86, 95, 93, 69, 87, 94, 74, 90, 76, 71, 87, 60, 72, 73, 77, 51, 66, 58, 63, 52.5, 67, 63, 78, 84, 69, 75, 77, 71, 82, 85, 82, 81.5, 94, 99, 97, 78, 93.5, 80, 92, 74, 71, 83, 70, 80, 86, 61, 77.6969111969112, 70, 83, 95, 82, 86, 83, 83, 82, 79, 101, 122, 100, 74, 70, 70, 70, 74, 73, 83, 66, 60, 66, 62, 60, 69, 67, 71, 68, 60, 68, 73, 66, 67, 72, 77, 67, 47, 68, 85.5, 84, 78, 89, 81, 61, 75, 99, 104, 83, 77.6969111969112, 62, 70, 91, 98, 103, 112, 105, 111, 109, 99, 110, 88, 82.5, 99, 81, 79, 72, 80, 75, 86, 77, 61, 56, 55, 66, 60, 71, 71, 74, 72, 54, 65, 74, 75, 76, 72, 69, 78.5, 67, 72, 63, 69, 87, 71, 71, 72.5, 75, 93, 89, 100, 96, 96, 101, 102, 70.5, 72, 74, 67, 68, 70, 65, 75.5, 72, 65, 80, 95, 94.5, 71, 84.5, 81, 78, 71, 66, 83, 85, 62, 73, 80, 69, 66, 63, 63, 69, 68, 78.5, 78, 78, 79, 67, 69, 82, 78, 61, 73.5, 70, 79, 81.5, 83, 90, 79, 99, 97, 95, 67, 79.5, 65, 80, 74, 70.5, 79, 78, 104, 77, 74, 87, 84, 94, 109, 91, 93.5, 95, 76, 72, 61, 57, 59, 70, 68, 82, 67, 69, 73, 76, 70, 57, 75, 63, 72, 64, 66, 70, 81, 68, 74, 72, 79, 84, 81, 69, 77, 74, 97, 103, 107, 88, 96, 101]

	print(len(mdnRnA), 0.7*len(mdnRnA), int(round(0.3*len(mdnRnA)))  )

	# New values 
	# It starts first week of July 2018 until 23th Sept 2018 (12 weeks)
	newValuesReal=[90, 106, 99, 104, 90, 80, 99, 100, 98, 85, 96, 84]

	################ ANN's en Keras ###################
	# 
	dataset = mdnRnA

	sample_size = 52 
	ahead = len(newValuesReal)
	dataset = np.asarray(dataset)
	nepochs=40
	
	assert 0 < sample_size < dataset.shape[0] 

	## Creacion de las muestras a partir del array normalizado ##
	X = np.atleast_3d(np.array([dataset[start:start + sample_size] 
		for start in range(0, dataset.shape[0]-sample_size)]))
	y = dataset[sample_size:]
	qf = np.atleast_3d([dataset[-sample_size:]]) 

	# Separamos en datos de entrenamiento y evaluacion
	#test_size = 52
	test_size = int(0.3*len(mdnRnA))

	trainX, testX = X[:-test_size], X[-test_size:]
	trainY, testY = y[:-test_size], y[-test_size:]


	nextSteps = np.empty((ahead+1,sample_size,1))
	nextSteps[0,:,:]= np.atleast_3d(np.array([dataset[start:start + sample_size] 
		for start in range(dataset.shape[0]-sample_size,dataset.shape[0]-sample_size+1)]))

	####### Creamos la estructura de la FFNN ###########
	####(usamos ReLU's como funciones de activacion)###
	
	# 2 capas ocultas con 64 y 32 neuronas, respectivamente
	neurons = [64, 32] 

	# Creamos la base del modelo
	model = Sequential() 

	# Ponemos una primera capa oculta con 64 neuronas
	model.add(Dense(neurons[0], activation='relu', 
			input_shape=(X.shape[1],X.shape[2])))

	print(model.layers[-1].output_shape)

	# Incorporamos una segunda capa oculta con 32 neuronas
	model.add(Dense(neurons[1], activation='relu'))
	print(model.layers[-1].output_shape)
	
	# Aplanamos los datos para reducir la dimensionalidad en la salida
	model.add(Flatten())

	# A\~nadimos la capa de salida de la red con activacion lineal
	model.add(Dense(1, activation='linear'))
	print(model.layers[-1].output_shape)

	# Compilamos el modelo usando el optimizador Adam
	model.compile(loss="mse", optimizer="adam") 
	#keras.utils.layer_utils.print_layer_shapes(model,input_shapes=(trainX.shape))

	# Entrenamos la red
	history = model.fit(trainX, trainY, epochs=nepochs, batch_size=10, verbose=0, validation_data=(testX, testY)) 

	### Pronostico de los datos de test ###
	pred = model.predict(testX)

	# Calcular ECM y EAM
	testScoreECM = mean_squared_error(testY, pred)
	print('ECM: %.4f' % (testScoreECM))

	testScoreEAM = mean_absolute_error(testY, pred)
	print('EAM: %.4f' % (testScoreEAM))

	''' predecir el futuro. '''
	newValues = np.zeros(ahead)
	temp=np.zeros(sample_size)

	for i in range(ahead):
		#print('ahead',i)
		#print('prediccion ', model.predict(nextSteps[None,i,:]), scaler.inverse_transform(model.predict(nextSteps[None,i,:])) )
		temp=nextSteps[i,1:,:]
		#print(temp, len(temp))
		temp = np.append(temp,model.predict(nextSteps[None,i,:]), axis=0)
		newValues[i] = model.predict(nextSteps[None,i,:])
		#print(temp, len(temp))

		#print(nextSteps[i,:,:])
		nextSteps[i+1,:,:]= temp
		#print(nextSteps[i+1,:,:])

	# Calcular ECM y EAM for ahead values
	#print('ECM ahead: %.4f' % (mean_squared_error(newValuesReal, newValues)))
	#print('EAM ahead: %.4f' % (mean_absolute_error(newValuesReal, newValues)))

	#print(model.output_shape)
	#print(model.summary())
	#print(model.get_config())


	startday = pd.datetime(2013, 7, 1)
	startdaypred = pd.datetime(2013, 7, 1) + 7*pd.Timedelta( len(mdnRnA)-len(pred), unit='D')
	startdayahead = pd.datetime(2013, 7, 1) + 7*pd.Timedelta( len(mdnRnA), unit='D')
	#print(startday,startdaypred,startdayahead)

	### Plotting ###
	# general plot
	fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,4))
	plt.figure(1)
	xaxis = ax.get_xaxis()
	#ax.xaxis.grid(b=True, which='minor', color='0.90', linewidth=0.6)
	ax.xaxis.set_major_locator(mdates.YearLocator())
	ax.xaxis.set_minor_locator(mdates.MonthLocator())
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
	#ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

	ax.plot(pd.date_range(startday, periods=len(mdnRnA), freq='W'), mdnRnA, linewidth=1, color='k') 
	ax.plot(pd.date_range(startdaypred, periods=len(pred), freq='W'), pred, linewidth=2, linestyle=':',color='k')
	#ax.plot(pd.date_range(startdayahead, periods=len(newValues), freq='W'), newValues, linestyle='--', color='b', linewidth=1 )
	#ax.plot(pd.date_range(startdayahead, periods=len(newValuesReal), freq='W'), newValuesReal, color='g', linewidth=1 )

	plt.suptitle('Weekly Predictions at LSC - Hall A')
	#plt.savefig('./prediction_ANN_weekly_D64D32_D1_e50_b10_ss52_ts52.eps')

	# ahead plot
	fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,4))# 
	plt.figure(2)
	xaxis = ax.get_xaxis()
	#ax.xaxis.grid(b=True, which='minor', color='0.90', linewidth=0.6)
	ax.xaxis.set_major_locator(mdates.YearLocator())
	ax.xaxis.set_minor_locator(mdates.MonthLocator())
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
	ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

	ax.plot(pd.date_range(startdayahead, periods=len(newValuesReal), freq='W'), newValuesReal, color='k', linewidth=1)
	ax.plot(pd.date_range(startdayahead, periods=len(newValues), freq='W'), newValues, linestyle=':', color='k', linewidth=2)

	plt.suptitle('Weekly Predictions at LSC - Hall A')
	#plt.savefig('./detailedprediction_ANN_weekly_D64D32_D1_e50_b10_ss52_ts52.eps')


	# summarize history for loss
	fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,6))# 6,6
	plt.figure(3)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	#ax.set_yscale("log")
	#plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	#plt.savefig('./loss_ANN_weekly_D64D32_D1_e50_b10_ss52_ts52.eps')

	#plt.show()


"""Invoking the main."""
main()
