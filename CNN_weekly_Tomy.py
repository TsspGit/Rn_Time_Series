#!/usr/bin/env python
# coding: utf-8

__author__ = ' AUTOR: Tomás Sánchez Sánchez-Pastor'

''' RNN Bidireccional en Keras '''


"""Imports: librerias"""
import os
import math
from math import sqrt
import sys
import numpy
import numpy as np
np.random.seed(0)
import random
import scipy.stats
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from time import time

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.initializers import glorot_uniform

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

def main(save_bool=True):

	# Medians Rn222 HallA (correcto 52 semanas X 5 años =259 valores)
	mdnRnA=[90, 79, 99, 117, 99, 99, 86, 95, 93, 69, 87, 94, 74, 90, 76, 71, 87, 60, 72, 73, 77, 51, 66, 58, 63, 52.5, 67, 63, 78, 84, 69, 75, 77, 71, 82, 85, 82, 81.5, 94, 99, 97, 78, 93.5, 80, 92, 74, 71, 83, 70, 80, 86, 61, 77.6969111969112, 70, 83, 95, 82, 86, 83, 83, 82, 79, 101, 122, 100, 74, 70, 70, 70, 74, 73, 83, 66, 60, 66, 62, 60, 69, 67, 71, 68, 60, 68, 73, 66, 67, 72, 77, 67, 47, 68, 85.5, 84, 78, 89, 81, 61, 75, 99, 104, 83, 77.6969111969112, 62, 70, 91, 98, 103, 112, 105, 111, 109, 99, 110, 88, 82.5, 99, 81, 79, 72, 80, 75, 86, 77, 61, 56, 55, 66, 60, 71, 71, 74, 72, 54, 65, 74, 75, 76, 72, 69, 78.5, 67, 72, 63, 69, 87, 71, 71, 72.5, 75, 93, 89, 100, 96, 96, 101, 102, 70.5, 72, 74, 67, 68, 70, 65, 75.5, 72, 65, 80, 95, 94.5, 71, 84.5, 81, 78, 71, 66, 83, 85, 62, 73, 80, 69, 66, 63, 63, 69, 68, 78.5, 78, 78, 79, 67, 69, 82, 78, 61, 73.5, 70, 79, 81.5, 83, 90, 79, 99, 97, 95, 67, 79.5, 65, 80, 74, 70.5, 79, 78, 104, 77, 74, 87, 84, 94, 109, 91, 93.5, 95, 76, 72, 61, 57, 59, 70, 68, 82, 67, 69, 73, 76, 70, 57, 75, 63, 72, 64, 66, 70, 81, 68, 74, 72, 79, 84, 81, 69, 77, 74, 97, 103, 107, 88, 96, 101]

	print("Length of the dataset: ",len(mdnRnA), "\nTrain_Size: ",0.7*len(mdnRnA), "\nTest_Size: ",int(round(0.3*len(mdnRnA)))  )

	# New values 
	# It starts first week of July 2018 until 23th Sept 2018 (12 weeks)
	newValuesReal=[90, 106, 99, 104, 90, 80, 99, 100, 98, 85, 96, 84]

	################ ANN's en Keras ###################

	dataset = mdnRnA


	sample_size = 52 
	ahead = len(newValuesReal)
	dataset = np.asarray(dataset)
	nepochs=40

	assert 0 < sample_size < dataset.shape[0] 

	# Aquí ha pasado a array la lista con los datos, saca por pantalla la longitud de los datos de entrenamiento y de los de test
	# y comprueba que las dimensiones del dataset son correctas. El algoritmo pasará por los datos 40 veces.
	#############################################################################################################################

	## Creacion de las muestras a partir del array normalizado ##
	X = np.atleast_3d(np.array([dataset[start:start + sample_size] 
	    for start in range(0, dataset.shape[0]-sample_size)]))
	# La lista comprimida crea ventanas de 52 datos empezando desde 0 hasta 52, lo convierte en array
	# y con atleast_3d lo convierte en una matriz tridimensional

	y = dataset[sample_size:]
	qf = np.atleast_3d([dataset[-sample_size:]]) 

	# Separamos en datos de entrenamiento y evaluacion
	#test_size = 52
	test_size = int(0.3*len(mdnRnA))

	trainX, testX = X[:-test_size], X[-test_size:]
	trainY, testY = y[:-test_size], y[-test_size:]
	print("trainX.shape = ", trainX.shape, "\ntrainY.shape = ", trainY.shape)

	nextSteps = np.empty((ahead+1,sample_size,1))
	nextSteps[0,:,:]= np.atleast_3d(np.array([dataset[start:start + sample_size] 
	    for start in range(dataset.shape[0]-sample_size,dataset.shape[0]-sample_size+1)]))

	# Convolutional model:
	neurons=[64, 32]
	model = Sequential()
	model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(52, 1), 
		        kernel_initializer=glorot_uniform(seed=0)))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(neurons[0], activation='relu'))
	model.add(Dense(neurons[1], activation='relu'))
	model.add(Dense(1))

	model.compile(loss="mse", optimizer="adam", metrics=["acc"])

	print(model.summary())

	history = model.fit(trainX, trainY, epochs=nepochs, batch_size=10, verbose=0, validation_data=(testX, testY)) 
	print(history.history.keys())

	pred = model.predict(testX)

	acc_train = np.average(history.history["acc"])
	acc_test = np.average(history.history["val_acc"])
	print("Train Accuracy: ", acc_train, "\nTest Accuracy:  ", acc_test)

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


	startday = pd.datetime(2013, 7, 1)
	startdaypred = pd.datetime(2013, 7, 1) + 7*pd.Timedelta( len(mdnRnA)-len(pred), unit='D')
	startdayahead = pd.datetime(2013, 7, 1) + 7*pd.Timedelta( len(mdnRnA), unit='D')
	#print(startday,startdaypred,startdayahead)

    
    
	### Plotting ###
    # general plot
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

    ax.plot(pd.date_range(startday, periods=len(mdnRnA), freq='W'), mdnRnA, linewidth=2, color='k', linestyle=':') 
    ax.plot(pd.date_range(startdaypred, periods=len(pred), freq='W'), pred, linewidth=2, linestyle='-',color='crimson')
    plt.xlabel('Time')
    ax.legend(['Data', 'CNN'], loc='upper left')
    plt.ylim([30, 140])

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
    axins = zoomed_inset_axes(ax, 1.7, loc='lower left', bbox_to_anchor=(640,140))
    axins.plot(pd.date_range(startday, periods=len(mdnRnA), freq='W'), mdnRnA, linewidth=2, color='k', linestyle=':') 
    axins.plot(pd.date_range(startdaypred, periods=len(pred), freq='W'), pred, linewidth=2, linestyle='-',color='crimson')
    axins.set_xlim('2017-01-01', '2018-06-17')
    axins.set_ylim(50, 110)

    plt.xticks(visible=False)
    plt.yticks(visible=False)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.suptitle('Weekly Fitting at LSC - Hall A')
	
	if save_bool:	
		plt.savefig('./fitting_CNN_weekly_D'+str(neurons[0])+'D'+str(neurons[1])+'_D1_e50_b10_ss52_ts52.eps')

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
	ax.plot(pd.date_range(startdayahead, periods=len(newValues), freq='W'), newValues, linestyle=':', color='b', linewidth=2)
	ax.legend(['Data', 'CNN'])
	plt.xlabel('Time')
	plt.suptitle('Weekly Predictions at LSC - Hall A')
	
	if save_bool:	
		plt.savefig('./detailedprediction_CNN_weekly_D'+str(neurons[0])+'D'+str(neurons[1])+'_D1_e50_b10_ss52_ts52.eps')


	# summarize history for loss
	fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,6))# 6,6
	plt.figure(3)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	#ax.set_yscale("log")
	#plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper right')
	
	if save_bool:		
		plt.savefig('./loss_CNN_weekly_D'+str(neurons[0])+'D'+str(neurons[1])+'_D1_e50_b10_ss52_ts52.eps')

	plt.show()

main()
