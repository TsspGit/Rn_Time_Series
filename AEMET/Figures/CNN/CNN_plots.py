__author__ = '@Tssp'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.stats.stats import pearsonr
from global_functions import read_datatxt
plt.rc('text',usetex=True)
plt.rc('font',family='serif')
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12

# Rn:
mdnRnA = np.loadtxt('../../../mdnRnA.txt', delimiter=',')
neuron = [64, 32]
startday = pd.datetime(2013, 7, 1)

## Plot:
CNN_loss = np.loadtxt('./CNN_Loss_Rn_{}_{}.txt'.format(neuron[0], neuron[1]), delimiter=',')
train_loss = CNN_loss[0]
test_loss = CNN_loss[1]
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,6))
plt.plot(train_loss)
plt.plot(test_loss)
plt.ylabel('loss', fontsize=14)
plt.xlabel('epoch', fontsize=14)
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
plt.savefig('./CNN_Loss_Rn_{}_{}.png'.format(neuron[0], neuron[1]))

## Plot 2:
pred = np.loadtxt('./CNN_Rn_{}_{}.txt'.format(neuron[0], neuron[1]), usecols=(0,))
startdaypred = pd.datetime(2013, 7, 1) + 7*pd.Timedelta( len(mdnRnA)-len(pred), unit='D')
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,4), constrained_layout=True)
xaxis = ax.get_xaxis()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax.plot(pd.date_range(startday, periods=len(mdnRnA), freq='W'), mdnRnA, 'k', alpha=0.7) 
ax.plot(pd.date_range(startdaypred, periods=len(pred), freq='W'), pred, linewidth=2, linestyle='-',color='crimson')
plt.xlabel('Dates', fontsize=16)
plt.ylabel(r'$^{222}Rn\ (Bq/m^3)$', fontsize=16)
ax.legend(['Data', 'CNN'], loc='upper left')
plt.ylim([30, 140])

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
axins = zoomed_inset_axes(ax, 1.7, loc='lower left', bbox_to_anchor=(2361,700))
axins.xaxis.set_major_locator(mdates.YearLocator())
axins.xaxis.set_minor_locator(mdates.MonthLocator())
axins.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axins.plot(pd.date_range(startday, periods=len(mdnRnA), freq='W'), mdnRnA, 'k', alpha=0.7) 
axins.plot(pd.date_range(startdaypred, periods=len(pred), freq='W'), pred, linewidth=2, linestyle='-',color='crimson')
axins.set_xlim('2017-10-05', '2019-07-21')
axins.set_ylim(50, 110)
plt.xticks(visible=True)
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.suptitle('Weekly Fitting at LSC - Hall A')
fig.savefig('./CNN_Rn_{}_{}.png'.format(neuron[0], neuron[1]), bbox_inches='tight', dpi=300)

# Rn + T

# Plot:
CNN_loss = np.loadtxt('./CNN_Loss_RnT_PMP_{}_{}.txt'.format(neuron[0], neuron[1]), delimiter=',')
train_loss = CNN_loss[0]
test_loss = CNN_loss[1]
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,6))
plt.plot(train_loss)
plt.plot(test_loss)
plt.ylabel('loss', fontsize=14)
plt.xlabel('epoch', fontsize=14)
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
plt.savefig('./CNN_Loss_RnT_PMP_{}_{}.png'.format(neuron[0], neuron[1]))

## Plot 2:
pred = np.loadtxt('./CNN_RnT_PMP_{}_{}.txt'.format(neuron[0], neuron[1]), usecols=(0,))
startdaypred = pd.datetime(2013, 7, 1) + 7*pd.Timedelta( len(mdnRnA)-len(pred), unit='D')
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,4), constrained_layout=True)
xaxis = ax.get_xaxis()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax.plot(pd.date_range(startday, periods=len(mdnRnA), freq='W'), mdnRnA, 'k', alpha=0.7) 
ax.plot(pd.date_range(startdaypred, periods=len(pred), freq='W'), pred, linewidth=2, linestyle='-',color='crimson')
plt.xlabel('Dates', fontsize=16)
plt.ylabel(r'$^{222}Rn\ (Bq/m^3)$', fontsize=16)
ax.legend(['Data', 'CNN(Rn + T)'], loc='upper left')
plt.ylim([30, 140])

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
axins = zoomed_inset_axes(ax, 1.7, loc='lower left', bbox_to_anchor=(2361,700))
axins.xaxis.set_major_locator(mdates.YearLocator())
axins.xaxis.set_minor_locator(mdates.MonthLocator())
axins.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axins.plot(pd.date_range(startday, periods=len(mdnRnA), freq='W'), mdnRnA, 'k', alpha=0.7) 
axins.plot(pd.date_range(startdaypred, periods=len(pred), freq='W'), pred, linewidth=2, linestyle='-',color='crimson')
axins.set_xlim('2017-10-05', '2019-07-21')
axins.set_ylim(50, 110)
plt.xticks(visible=True)
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.suptitle('Weekly Fitting at LSC - Hall A')
fig.savefig('./CNN_RnT_PMP_{}_{}.png'.format(neuron[0], neuron[1]), bbox_inches='tight', dpi=300)
plt.show()